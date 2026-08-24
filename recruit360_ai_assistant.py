
import os, json, re, time
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from langchain.tools import tool
from langchain.agents import create_agent

PROJECT="direct-tribute-502305-q5"; DATASET="recruit360"; LOCATION="us-central1"; MODEL="gemini-2.5-flash"
def T(n): return f"`{PROJECT}.{DATASET}.{n}`"
st.set_page_config(page_title="Recruit360 AI Assistant", page_icon="🤖", layout="wide")

def _creds():
    try:
        if "gcp_service_account" in st.secrets:
            with open("/tmp/sa.json","w") as f: json.dump(dict(st.secrets["gcp_service_account"]),f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/tmp/sa.json"
    except Exception: pass
_creds()
try: MAPS_KEY = st.secrets.get("MAPS_API_KEY", "")
except Exception: MAPS_KEY = ""
# --- Document AI config (fill these from your processor page) ---
DOCAI_LOCATION  = "us"                    # region of your processor, e.g. "us" or "eu"
DOCAI_PROCESSOR = "22d406f8def70c29"       # the Processor ID from the console
import requests, math
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

def _invoke(prompt, tries=4):
    for i in range(tries):
        try: return llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "exhausted" in str(e).lower(): time.sleep(4*(i+1)); continue
            raise
    return llm.invoke(prompt)

# ---- QUERY OPTIMIZATION: result cache + BigQuery cost cap (faster + cheaper) ----
_SQL_CACHE={}
def run_sql(sql):
    if sql in _SQL_CACHE:                         # repeat question -> instant, no re-scan, no cost
        return _SQL_CACHE[sql]
    cfg=bigquery.QueryJobConfig(use_query_cache=True, maximum_bytes_billed=200_000_000)  # 200MB cost cap
    df=bq.query(sql, job_config=cfg).to_dataframe()
    if not df.empty:                              # never cache an empty/failed result
        _SQL_CACHE[sql]=df
    return df

ART={}
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
-- IMPORTANT: "visa rejected" means candidates.visa_status = 'VISA_REJECTED'.
-- Count rejected candidates with: SELECT COUNT(*) FROM candidates WHERE visa_status='VISA_REJECTED'.
candidates(candidate_id, full_name, email, origin_city, destination_country, destination_employer,
  role, recruiter, csr_owner, training_centre, visa_agency, visa_status, deposit_amount, currency,
  urgency_score, created_at)
visa_workflows(visa_id, candidate_id, visa_agency, jurisdiction, submitted_date, decision,
  rejection_codes, retry_count, decision_date)
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

@tool
def query_recruitment_data(question: str) -> str:
    """Answer ANY analytical or lookup question about Recruit360 — candidates, visas, billing,
    training, jobs, placements. Handles counts, totals, filters, joins and lookups. Main data tool."""
    _trace("Conversational/Reporting")
    prompt=(f"Write ONE efficient BigQuery SELECT (only SQL, no fences).\n"
            f"Select ONLY the columns needed (never SELECT *). For list/detail questions add LIMIT 100. "
            f"Use COUNT/SUM/AVG for totals. When the user filters by a role, city, skill or status, "
            f"use a WHERE clause with LOWER(column) LIKE LOWER('%value%') so matching is case-insensitive.\n"
            f"{SCHEMA}\nQuestion: {question}\nSQL:")
    sql=re.sub(r"^```(?:sql)?|```$","",_txt(_invoke(prompt)),flags=re.I).strip()
    if not _readonly(sql): return f"Blocked (read-only).\n{sql}"
    try: df=run_sql(sql)
    except Exception as e: return f"Query failed: {e}\nSQL:\n{sql}"
    ART["table"]=df
    if df.empty:
        return (f"No records match this request in the database. "
                f"There are zero results for: {question}. "
                f"Do not invent any — the correct answer is that none were found.\n\nSQL:\n{sql}")
    return f"SQL:\n{sql}\n\nResult:\n{df.head(30).to_string(index=False)}"

@tool
def visa_fix_it(candidate_id: str) -> str:
    """For a candidate whose visa was REJECTED, read their rejection codes and produce a clear
    step-by-step remediation checklist. Input: a candidate_id like 'C2013'."""
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
        f"Write a clear, numbered remediation checklist a recruiter can act on to fix and re-apply."))
    ART["table"]=df
    return f"Rejection codes for {cid}: {', '.join(codes)}\n\nRemediation checklist:\n{ans}"

@tool
def urgency_watch(top_n: int = 10) -> str:
    """List the highest-urgency candidates and AWOL risks (visa_status NOT_REPORTED), with reasons."""
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

@tool
def semantic_search_candidates(description: str, top_k: int = 5) -> str:
    """Find candidates by MEANING (not keywords) using Vertex embeddings + BigQuery vector search.
    Input: a natural description, e.g. 'experienced backend engineer going to Germany'."""
    _trace("Semantic Search")
    sql=f"""
    SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role,
           c.destination_country, c.visa_status, h.distance
    FROM VECTOR_SEARCH(
      TABLE {T('candidate_embeddings')}, 'embedding',
      (SELECT ml_generate_embedding_result AS embedding
       FROM ML.GENERATE_EMBEDDING(MODEL {T('text_embedder')}, (SELECT @q AS content))),
      top_k => {int(top_k)}) AS h
    JOIN {T('candidates')} c ON c.candidate_id = h.base.candidate_id
    ORDER BY h.distance"""
    try:
        df=bq.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("q","STRING",description)])).to_dataframe()
    except Exception as e:
        return f"Semantic search error: {e}"
    # Reject weak matches: vector distance above threshold means "not really relevant".
    THRESHOLD = 0.74
    if not df.empty:
        df = df[df["distance"] <= THRESHOLD]
    if df.empty:
        return (f"No candidates in the database closely match '{description}'. "
                f"I won't guess — there is no strong match for this request.")
    ART["table"]=df
    return f"Candidates matching '{description}' by meaning:\n{df.to_string(index=False)}"

@tool
def predict_placement(top_n: int = 10) -> str:
    """Predict which candidates are MOST LIKELY to be placed, using the BigQuery ML model.
    Use when asked about placement likelihood, best prospects, or who to prioritise."""
    _trace("Predict-Score")
    sql=f"""
    SELECT candidate_id,
           ROUND((SELECT prob FROM UNNEST(predicted_is_placed_probs) WHERE label=1),3) AS placement_probability
    FROM ML.PREDICT(MODEL {T('placement_predictor')},
      (SELECT * FROM {T('candidates')}))
    ORDER BY placement_probability DESC
    LIMIT {int(top_n)}"""
    try:
        df=bq.query(sql).to_dataframe()
        df=df.merge(bq.query(f"SELECT candidate_id, full_name, role, destination_country FROM {T('candidates')}").to_dataframe(),
                    on="candidate_id", how="left")
    except Exception as e:
        return f"Prediction error: {e}"
    ART["table"]=df
    return f"Candidates most likely to be placed:\n{df.to_string(index=False)}"

_GEO_CACHE={}
def _geocode(place):
    if place in _GEO_CACHE: return _GEO_CACHE[place]
    r=requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                   params={"address":place+", India","key":MAPS_KEY}, timeout=10).json()
    if r.get("status")=="OK":
        loc=r["results"][0]["geometry"]["location"]; _GEO_CACHE[place]=(loc["lat"],loc["lng"]); return _GEO_CACHE[place]
    _GEO_CACHE[place]=None; return None
def _haversine(a,b):
    R=6371; la1,lo1=map(math.radians,a); la2,lo2=map(math.radians,b)
    d=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(d))

@tool
def candidates_near_city(city: str, radius_km: float = 300) -> str:
    """Find candidates whose origin city is near a given city, using Google Maps geocoding.
    Input: a city name (e.g. 'Bengaluru'). Returns candidates within radius_km, nearest first."""
    _trace("Location/Maps")
    if not MAPS_KEY: return "Maps API key not configured (add MAPS_API_KEY to secrets)."
    target=_geocode(city)
    if not target: return f"Could not locate '{city}'."
    try:
        cities=bq.query(f"SELECT DISTINCT origin_city FROM {T('candidates')} WHERE origin_city IS NOT NULL").to_dataframe()
    except Exception as e: return f"Query failed: {e}"
    near=[]
    for oc in cities["origin_city"]:
        p=_geocode(oc)
        if p:
            dist=_haversine(target,p)
            if dist<=radius_km: near.append((oc, round(dist,1)))
    if not near: return f"No candidate origin cities within {radius_km} km of {city}."
    near.sort(key=lambda x:x[1])
    cities_in=[c for c,_ in near]
    df=bq.query(f"""SELECT candidate_id, full_name, role, origin_city, destination_country, visa_status
                    FROM {T('candidates')} WHERE origin_city IN UNNEST(@c) LIMIT 50""",
                job_config=bigquery.QueryJobConfig(
                  query_parameters=[bigquery.ArrayQueryParameter("c","STRING",cities_in)])).to_dataframe()
    dmap=dict(near); df["distance_km"]=df["origin_city"].map(dmap)
    df=df.sort_values("distance_km")
    ART["table"]=df
    return f"Candidates near {city} (within {radius_km} km):\n{df.to_string(index=False)}"

@tool
def match_candidates_to_job(job_title: str, destination_country: str = "", skills: str = "", top_k: int = 5) -> str:
    """STEP 1 of recruitment - SOURCING. Given a JOB REQUIREMENT (job title, optional destination
    and skills), find the best-matching candidates for that job using semantic search.
    Use this when a recruiter asks 'find candidates for this job' or 'who matches this role'."""
    _trace("Job-Candidate Match")
    # build a job-requirement description, then reuse the embedding + vector search
    jobdesc = f"{job_title}"
    if skills: jobdesc += f" with skills {skills}"
    if destination_country: jobdesc += f" going to {destination_country}"
    sql=f"""
    SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role,
           c.destination_country, c.visa_status, h.distance
    FROM VECTOR_SEARCH(
      TABLE {T('candidate_embeddings')}, 'embedding',
      (SELECT ml_generate_embedding_result AS embedding
       FROM ML.GENERATE_EMBEDDING(MODEL {T('text_embedder')}, (SELECT @q AS content))),
      top_k => {int(top_k)}) AS h
    JOIN {T('candidates')} c ON c.candidate_id = h.base.candidate_id"""
    if destination_country:
        sql += " WHERE LOWER(c.destination_country) = LOWER(@dest)"
    sql += " ORDER BY h.distance"
    try:
        params=[bigquery.ScalarQueryParameter("q","STRING",jobdesc)]
        if destination_country:
            params.append(bigquery.ScalarQueryParameter("dest","STRING",destination_country))
        df=bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    except Exception as e:
        return f"Matching error: {e}"
    # Reject weak matches so we never present unrelated people as a match.
    THRESHOLD = 0.74
    if not df.empty:
        df = df[df["distance"] <= THRESHOLD]
    if df.empty:
        return (f"No matching candidates found for '{job_title}'"
                + (f" going to {destination_country}" if destination_country else "")
                + ". There is no candidate in the database that fits this requirement.")
    ART["table"]=df
    lines=[f"Best-matching candidates for the job '{job_title}'" + (f" ({destination_country})" if destination_country else "") + ":"]
    for _,r in df.iterrows():
        lines.append(f"- {r['full_name']} ({r['candidate_id']}) - {r['role']}, going to {r['destination_country']}, visa {r['visa_status']}")
    return "\n".join(lines)

TOOLS=[query_recruitment_data, visa_fix_it, urgency_watch, semantic_search_candidates, predict_placement, candidates_near_city, match_candidates_to_job]
SYSTEM=("You are the Recruit360 AI Assistant for CSRs and recruiters. "
        "Use query_recruitment_data for any factual data question (counts, lists, filters by role/city/status). "
        "Use visa_fix_it to fix a rejected visa. Use urgency_watch for urgent/at-risk cases. "
        "Use semantic_search_candidates or match_candidates_to_job to find candidates by description. "
        "CRITICAL RULES: "
        "1. NEVER invent, guess, or fabricate candidates. Only report what the tools return. "
        "2. If a tool returns 'No candidates' or 'no strong match', you MUST tell the user clearly that there are NO matching candidates. Do NOT substitute other people. "
        "3. If the user asks for a specific role (e.g. 'Java developer') and none exist, say plainly: there are no candidates with that role in the database. "
        "4. When a specific attribute is requested (a role, a city, a skill), prefer query_recruitment_data with a precise WHERE filter over semantic search, so results are exact. "
        "5. Data is read-only. Report tool output faithfully and never add candidates that were not returned.")
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
            if ("429" in str(e) or "exhausted" in str(e).lower()) and i<2: time.sleep(5); continue
            raise

# ---------------- UI (original purple) ----------------
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
    for t in ["Conversational + Reporting","Visa Fix-It","Urgency / Risk Watch","Semantic Search","Predict-Score","Location / Maps","Job-Candidate Match"]:
        st.markdown(f'<div class="tool">{t}</div>',unsafe_allow_html=True)
    st.markdown("---"); st.markdown("**💡 Try these**")
    for e in ["How many candidates are visa rejected?","Total billing amount by billing domain",
              "Top 5 destination countries by candidate count","Fix the visa for candidate C2013",
              "Which candidates are most urgent right now?","How many placements and total fees?","Find candidates like an experienced engineer going to Germany","Which candidates are most likely to be placed?","Candidates near Bengaluru","Find candidates for a QA Engineer job going to Sweden"]:
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


# ================= DOCUMENT AI — extract candidate from a document =================
st.markdown("---")
with st.expander("📄 Document AI — extract candidate details from a document image"):
    up = st.file_uploader("Upload a résumé / ID / visa document (PNG, JPG, PDF)",
                          type=["png","jpg","jpeg","pdf"], key="docai_up")
    if up and st.button("Extract details", key="docai_extract"):
        try:
            from google.cloud import documentai
            client = documentai.DocumentProcessorServiceClient(
                client_options={"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"})
            name = f"projects/{PROJECT}/locations/{DOCAI_LOCATION}/processors/{DOCAI_PROCESSOR}"
            mime = up.type or "application/pdf"
            raw = documentai.RawDocument(content=up.getvalue(), mime_type=mime)
            result = client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw))
            text = result.document.text
            # ask Gemini to structure it
            prompt = ("Extract candidate details from this document text as JSON with keys: "
                      "full_name, email, phone, role, origin_city, destination_country, passport_number. "
                      "Use empty string if a field is missing. Return ONLY JSON.\n\n" + text[:6000])
            raw_json = _txt(_invoke(prompt))
            raw_json = re.sub(r"^```(?:json)?|```$", "", raw_json.strip(), flags=re.I).strip()
            st.session_state["docai_fields"] = json.loads(raw_json)
            st.success("Extracted — review the fields below, then Save.")
        except Exception as e:
            st.error(f"Document AI error: {e}")

    f = st.session_state.get("docai_fields")
    if f:
        st.markdown("**Review & edit the extracted fields:**")
        c1, c2 = st.columns(2)
        with c1:
            f["full_name"] = st.text_input("Full name", f.get("full_name",""))
            f["email"]     = st.text_input("Email", f.get("email",""))
            f["phone"]     = st.text_input("Phone", f.get("phone",""))
            f["role"]      = st.text_input("Role", f.get("role",""))
        with c2:
            f["origin_city"]         = st.text_input("Origin city", f.get("origin_city",""))
            f["destination_country"] = st.text_input("Destination country", f.get("destination_country",""))
            f["passport_number"]     = st.text_input("Passport number", f.get("passport_number",""))
        if st.button("💾 Save candidate to BigQuery", key="docai_save"):
            try:
                import uuid
                cid = "C" + str(uuid.uuid4().int)[:5]
                row = {"candidate_id": cid, "full_name": f["full_name"], "email": f["email"],
                       "origin_city": f["origin_city"], "destination_country": f["destination_country"],
                       "role": f["role"], "visa_status": "INTAKE_PENDING",
                       "passport_number": f["passport_number"]}
                errors = bq.insert_rows_json(f"{PROJECT}.{DATASET}.candidates", [row])
                if errors: st.error(f"Save failed: {errors}")
                else:
                    st.success(f"Saved candidate {cid} ({f['full_name']}) to BigQuery.")
                    st.session_state["docai_fields"] = None
            except Exception as e:
                st.error(f"Save error: {e}")
