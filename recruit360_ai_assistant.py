
import os, json, re
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from langchain.tools import tool
from langchain.agents import create_agent

PROJECT="direct-tribute-502305-q5"; DATASET="recruit360"; LOCATION="us-central1"; MODEL="gemini-2.5-flash"
def T(n): return f"`{PROJECT}.{DATASET}.{n}`"

st.set_page_config(page_title="Recruit360 AI Assistant", page_icon="🤖", layout="wide")

# ---------------- auth ----------------
def _creds():
    try:
        if "gcp_service_account" in st.secrets:
            with open("/tmp/sa.json","w") as f: json.dump(dict(st.secrets["gcp_service_account"]),f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/tmp/sa.json"
    except Exception: pass
_creds()
import vertexai
try: vertexai.init(project=PROJECT, location=LOCATION)
except Exception as e: st.warning(f"Vertex init: {e}")

@st.cache_resource(show_spinner=False)
def get_bq(): return bigquery.Client(project=PROJECT)
@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_google_vertexai import ChatVertexAI
    return ChatVertexAI(model=MODEL, project=PROJECT, location=LOCATION, temperature=0)
bq=get_bq(); llm=get_llm()

import time
def _invoke(prompt, tries=4):
    """Call Vertex with gentle retry so occasional 429 rate-limits don't reach the user."""
    for i in range(tries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "exhausted" in str(e).lower():
                time.sleep(4*(i+1)); continue
            raise
    return llm.invoke(prompt)

ART={}                                   # tools run in threads -> plain dict, bridged to session
if "art" not in st.session_state: st.session_state["art"]={}
def _txt(m):
    c=getattr(m,"content",m)
    if isinstance(c,str): return c.strip()
    return "".join(b.get("text","") for b in c if isinstance(b,dict)).strip()
def _trace(n): ART.setdefault("trace",[]).append(n)
def _readonly(sql):
    l=sql.lower().strip()
    return l.startswith("select") and not re.search(r"\b(insert|update|delete|drop|create|alter|merge|truncate)\b",l)

SCHEMA=f"""Tables in `{PROJECT}.{DATASET}` (join on ids):
candidates(candidate_id, full_name, email, origin_city, destination_country, destination_employer,
  role, recruiter, csr_owner, training_centre, visa_agency, visa_status, deposit_amount, currency,
  urgency_score, created_at)  -- visa_status: INTAKE_PENDING, TRAINING_IN_PROGRESS, VISA_SUBMITTED,
  -- VISA_APPROVED, VISA_REJECTED, REMEDIATION_IN_PROGRESS, TRAVEL_CONFIRMED, ARRIVED, REPORTED,
  -- NOT_REPORTED, PLACEMENT_ACTIVE
visa_workflows(visa_id, candidate_id, visa_agency, jurisdiction, submitted_date, decision,
  rejection_codes, retry_count, decision_date)  -- decision: PENDING/APPROVED/REJECTED
billing_schedules(billing_id, candidate_id, billing_domain, amount, currency, status, due_date)
training_centres(centre_id, name, city, country)
jobs(job_id, title, client, department, location, status, openings, recruiter, created_date)
placements(placement_id, candidate_id, job_id, placement_date, fee_eur, status)"""

REJECTION_FIXES={
 "R-01":"Incomplete application form — complete all mandatory fields and resubmit.",
 "R-02":"Invalid/expired passport — renew passport (min 6 months validity) and attach a clear copy.",
 "R-03":"Insufficient financial proof — provide bank statements or a sponsorship letter meeting the threshold.",
 "R-04":"Missing employer sponsorship — obtain a signed sponsorship letter and employment contract.",
 "R-05":"Photograph does not meet spec — submit a biometric photo per the embassy specification.",
 "R-06":"Missing/invalid qualifications — attach attested degree/certificates with certified translation.",
 "R-07":"Medical certificate missing — complete the panel medical and attach the report.",
 "R-08":"Travel/health insurance missing — purchase a compliant policy and attach it.",
 "R-09":"Inconsistent personal details — correct mismatched details across all documents.",
 "R-10":"Interview/extra docs required — schedule the embassy interview or provide the requested documents."}

# ============ AGENT 1 — Conversational + Reporting (text-to-SQL) ============
@tool
def query_recruitment_data(question: str) -> str:
    """Answer ANY analytical or lookup question about Recruit360 — candidates, visas, billing,
    training, jobs, placements. Handles counts, totals, filters, joins and lookups. Input: a
    plain-English question. This is the main data tool."""
    _trace("Conversational/Reporting")
    prompt=f"Write ONE BigQuery SELECT (only SQL, no fences) answering the question.\n{SCHEMA}\nQuestion: {question}\nSQL:"
    sql=re.sub(r"^```(?:sql)?|```$","",_txt(_invoke(prompt)),flags=re.I).strip()
    if not _readonly(sql): return f"Blocked (read-only).\n{sql}"
    try: df=bq.query(sql).to_dataframe()
    except Exception as e: return f"Query failed: {e}\nSQL:\n{sql}"
    ART["table"]=df
    return f"SQL:\n{sql}\n\nResult:\n{df.head(30).to_string(index=False)}"

# ============ AGENT 2 — Visa Fix-It ============
@tool
def visa_fix_it(candidate_id: str) -> str:
    """For a candidate whose visa was REJECTED, read their rejection codes and produce a clear,
    step-by-step remediation checklist to fix and re-apply. Input: a candidate_id like 'C2013'."""
    _trace("Visa Fix-It")
    cid=candidate_id.strip().upper()
    try:
        df=bq.query(f"SELECT candidate_id, decision, rejection_codes, retry_count FROM {T('visa_workflows')} "
                    f"WHERE candidate_id=@c AND decision='REJECTED'",
                    job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("c","STRING",cid)])).to_dataframe()
    except Exception as e: return f"Lookup failed: {e}"
    if df.empty: return f"No rejected visa found for {cid} (nothing to remediate)."
    codes=[c for cell in df["rejection_codes"].dropna() for c in str(cell).split(";") if c]
    mapped="\n".join(f"- {c}: {REJECTION_FIXES.get(c,'Refer to embassy guidance.')}" for c in codes)
    ans=_txt(_invoke(
        f"You are a visa remediation assistant. A candidate's visa was rejected with these issues:\n{mapped}\n\n"
        f"Write a clear, numbered remediation checklist a recruiter can act on to fix and re-apply. "
        f"Keep it practical and specific."))
    ART["table"]=df
    return f"Rejection codes for {cid}: {', '.join(codes)}\n\nRemediation checklist:\n{ans}"

# ============ AGENT 3 — Urgency / Risk Watch ============
@tool
def urgency_watch(top_n: int = 10) -> str:
    """List the highest-urgency candidates and any AWOL risks (visa_status NOT_REPORTED), with reasons.
    Use when asked about urgent cases, risks, or what needs attention."""
    _trace("Urgency/Risk")
    try:
        df=bq.query(f"""SELECT candidate_id, full_name, role, destination_country, visa_status, urgency_score
                        FROM {T('candidates')}
                        WHERE visa_status IN ('VISA_REJECTED','REMEDIATION_IN_PROGRESS','NOT_REPORTED')
                           OR urgency_score >= 75
                        ORDER BY urgency_score DESC LIMIT {int(top_n)}""").to_dataframe()
    except Exception as e: return f"Query failed: {e}"
    if df.empty: return "No high-urgency candidates right now."
    ART["table"]=df
    return f"Top urgent / at-risk candidates:\n{df.to_string(index=False)}"

TOOLS=[query_recruitment_data, visa_fix_it, urgency_watch]
SYSTEM=("You are the Recruit360 AI Assistant for CSRs and recruiters. "
        "Use query_recruitment_data for any data question (counts, totals, filters, lookups). "
        "Use visa_fix_it when asked how to fix a rejected visa for a candidate. "
        "Use urgency_watch for urgent/at-risk cases. Never invent data; data is read-only.")

@st.cache_resource(show_spinner=False)
def get_agent(): return create_agent(model=get_llm(), tools=TOOLS, system_prompt=SYSTEM)
agent=get_agent()
def ask(q):
    ART.clear(); ART["trace"]=[]
    for i in range(3):
        try:
            r=agent.invoke({"messages":[{"role":"user","content":q}]})
            st.session_state["art"]=dict(ART)
            return _txt(r["messages"][-1])
        except Exception as e:
            if ("429" in str(e) or "exhausted" in str(e).lower()) and i<2:
                time.sleep(5); continue
            raise

# ---------------- UI ----------------
st.markdown("""<style>
.stApp{background:linear-gradient(180deg,#0e1117,#131a2b);}
.hero{padding:18px 26px;border-radius:14px;margin-bottom:8px;color:#fff;
 background:linear-gradient(100deg,#0b1b4d,#6C4FE0 55%,#00c6ff);}
.hero h1{margin:0;font-size:26px;font-weight:800;} .hero p{margin:3px 0 0;opacity:.92;font-size:14px;}
.chip{display:inline-block;padding:3px 10px;border-radius:999px;background:rgba(255,255,255,.15);
 color:#fff;font-size:11px;margin-right:6px;}
.tool{display:inline-block;padding:2px 9px;margin:3px 0;border-radius:8px;background:#1f2740;color:#8ab4f8;
 font-size:12px;border:1px solid #2b3550;}
</style>""",unsafe_allow_html=True)
st.markdown("""<div class="hero"><h1>🤖 Recruit360 AI Assistant</h1>
<p>One assistant, multiple AI agents — on Vertex AI + BigQuery. Ask anything about candidates, visas, billing, jobs and placements.</p>
<div style="margin-top:8px"><span class="chip">Vertex AI brain</span><span class="chip">BigQuery</span>
<span class="chip">LangChain agents</span><span class="chip">Read-only guardrails</span></div></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.subheader("🧠 AI agents in this assistant")
    for t in ["Conversational + Reporting","Visa Fix-It","Urgency / Risk Watch"]:
        st.markdown(f'<div class="tool">{t}</div>',unsafe_allow_html=True)
    st.markdown("---"); st.markdown("**💡 Try these**")
    ex=["How many candidates are visa rejected?",
        "Total billing amount by billing domain",
        "Top 5 destination countries by candidate count",
        "Fix the visa for candidate C2013",
        "Which candidates are most urgent right now?",
        "How many placements and total fees?"]
    for e in ex:
        if st.button(e,use_container_width=True): st.session_state.pending=e

if "hist" not in st.session_state: st.session_state.hist=[]
for role,txt in st.session_state.hist:
    with st.chat_message(role): st.markdown(txt)
prompt=st.chat_input("Ask the Recruit360 assistant…")
if st.session_state.get("pending"): prompt=st.session_state.pending; st.session_state.pending=None
if prompt:
    st.session_state.hist.append(("user",prompt))
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking across the AI agents…"):
            try: ans=ask(prompt)
            except Exception as e: ans=f"Something went wrong: {e}"
        st.markdown(ans)
        tr=st.session_state.get("art",{}).get("trace",[])
        if tr: st.caption("🧠 agent used: "+" → ".join(tr))
    st.session_state.hist.append(("assistant",ans))

art=st.session_state.get("art",{})
if isinstance(art.get("table"),pd.DataFrame) and not art["table"].empty:
    st.markdown("#### 📋 Result data"); st.dataframe(art["table"],use_container_width=True,height=320)
st.caption("Recruit360 AI Assistant · GCP (Vertex AI + BigQuery + LangChain) · rebuild of the reference, AI-first.")
