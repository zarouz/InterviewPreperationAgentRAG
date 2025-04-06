# utils.py
import os
import re
import psycopg2
import pdfplumber
from sentence_transformers import SentenceTransformer
import logging
import nltk
import ssl
from nltk.corpus import stopwords

# Import constants from config
from config import (
    DB_NAME, DB_USER, DB_PASSWORD, DB_HOST,
    EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K, RETRIEVAL_SIMILARITY_THRESHOLD,
    MAX_CONTEXT_LENGTH,
    NUM_QUESTIONS # <-- ADD THIS IMPORT
)

logger = logging.getLogger(__name__)

# --- NLTK Download ---
def download_nltk_data(resource_id, resource_type='corpora'):
    try:
        nltk.data.find(f'{resource_type}/{resource_id}')
        logger.debug(f"NLTK resource '{resource_id}' already present.")
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource_id}")
        try:
            _original_ssl_context = ssl.create_default_context()
            unverified_context = ssl._create_unverified_context()
            ssl._create_default_https_context = lambda: unverified_context
            nltk.download(resource_id, quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK resource '{resource_id}': {e}")
        finally:
            ssl._create_default_https_context = _original_ssl_context

# utils.py - near the top where embedding_model is initialized

# --- Embedding Model Initialization ---
embedding_model = None # Initialize as None
try:
    logger.info(f"Attempting to load embedding model: {EMBEDDING_MODEL_NAME}...")
    # Ensure SentenceTransformer is imported correctly
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    # --- THIS IS THE CRITICAL PART ---
    # Log the exception WITH traceback information
    logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    # --- Make sure exc_info=True is present and saved ---
    embedding_model = None # Keep it None on failure
    # Optional: print to console as a backup if logging fails
    print(f"!!! ERROR: Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    import traceback
    print("!!! TRACEBACK:")
    traceback.print_exc()
    print("!!! END TRACEBACK")
    # raise RuntimeError(f"Failed to load embedding model: {e}") from e # Keep commented out for now
    
# --- PDF Parsing ---
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        logger.error(f"Resume PDF not found at: {pdf_path}")
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                 try:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if page_text: full_text += page_text + "\n"
                 except Exception as page_err:
                     logger.warning(f"Could not extract text from page {i+1} in {pdf_path}: {page_err}")
        logger.info(f"Successfully extracted text from PDF: {pdf_path}")
        full_text = re.sub(r' +', ' ', full_text)
        full_text = re.sub(r'\n+', '\n', full_text)
        return full_text.strip()
    except Exception as e:
        logger.error(f"Error opening or processing PDF {pdf_path}: {e}")
        raise

# --- Text Preprocessing ---
def preprocess_query(query):
    query = re.sub(r'[^\w\s.-]', ' ', query).lower()
    query = re.sub(r'\s+', ' ', query).strip()
    return query

# --- RAG Retrieval ---
def retrieve_similar_documents(query, top_k=RETRIEVAL_TOP_K, similarity_threshold=RETRIEVAL_SIMILARITY_THRESHOLD):
    """Retrieve similar document chunks from the database."""
    if not embedding_model:
        logger.error("Embedding model not initialized, cannot retrieve documents.")
        return []

    processed_query = preprocess_query(query)
    if not processed_query:
        logger.warning("Empty query after preprocessing, skipping retrieval.")
        return []
    try:
        query_embedding = embedding_model.encode(processed_query)
    except Exception as e:
        logger.error(f"Failed to encode query '{query}': {e}")
        return []

    conn = None
    cursor = None
    documents = []
    try:
        conn_string = f"dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}' host='{DB_HOST}'"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        sql = """
        SELECT content, 1 - (embedding <=> %s::vector) as similarity_score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        cursor.execute(sql, (query_embedding.tolist(), query_embedding.tolist(), top_k * 3))
        rows = cursor.fetchall()
        documents = [{"content": row[0], "score": row[1]} for row in rows if row[1] >= similarity_threshold]
        documents = sorted(documents, key=lambda x: x["score"], reverse=True)[:top_k]
        if documents:
             logger.info(f"Retrieved {len(documents)} chunks for query '{query}' with threshold {similarity_threshold}.")
    except psycopg2.OperationalError as db_conn_err:
         logger.error(f"Database connection error: {db_conn_err}")
         # Return empty list or raise error depending on desired handling
    except psycopg2.Error as db_err:
        logger.error(f"Database query error during retrieval: {db_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during retrieval: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
    return documents

def format_rag_context(documents, max_length=MAX_CONTEXT_LENGTH):
    """Formats retrieved documents into a single context string for the LLM."""
    if not documents: return ""
    context = "CONTEXT FROM KNOWLEDGE BASE:\n\n"
    total_length = len(context)
    added_content = set()
    sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    for doc in sorted_docs:
        content = doc.get("content", "").strip()
        score = doc.get("score", 0)
        if not content or len(content) < 10 or content in added_content: continue
        doc_text = f"[Source Relevance: {score:.2f}] {content}\n\n"
        doc_length = len(doc_text)
        if total_length + doc_length <= max_length:
            context += doc_text
            total_length += doc_length
            added_content.add(content)
        else:
            logger.info(f"Context length limit ({max_length} chars) reached. Added {len(added_content)} documents.")
            break
    logger.info(f"Formatted context final length: {len(context)} characters.")
    return context.strip()

# --- Rule-Based Skill Extraction ---
# (SKILL_CATEGORIES and extract_skills_from_text function remain the same as the last version)
SKILL_CATEGORIES = {
    "Programming Languages": ["python", "java", "c++", "sql", "javascript", "bash", "shell", "powershell", "c#", "golang", "rust", "scala", "php", "typescript", "ruby", "perl", "haskell", "objective-c", "swift", "dart", "elixir", "cobol", "fortran", "lisp", "matlab", "assembly", "r", "groovy"],
    "Databases": ["database", "sql", "nosql", "postgresql", "mysql", "sql server", "oracle", "mongodb", "redis", "cassandra", "sqlite", "sqlalchemy", "database administration", "dba", "indexing", "index", "normalization", "acid", "acid properties", "query optimization", "query plan", "execution plan", "backup", "recovery", "replication", "dbms", "er diagrams", "stored procedures", "triggers", "data modeling", "data integrity", "data warehouse", "bigquery", "db tuning", "partitioning", "json sql", "pl/sql", "t-sql", "db security", "data migration", "nosql design", "graph database", "in-memory database", "sql injection", "database performance", "distributed database", "concurrency control", "locking", "relational model"],
    "Web Frameworks": ["flask", "django", "fastapi", "react", "angular", "vue", "node.js", "node", "spring", "express", "ember.js", "backbone.js", "svelte", "laravel", "ruby on rails", "asp.net", "meteor", "symfony", "play framework"],
    "Operating Systems": ["linux", "unix", "windows", "operating system", "kernel", "process", "thread", "scheduling", "memory management", "file system", "macos", "ios", "android os", "bsd", "solaris", "real-time os", "embedded os", "virtualization", "bootloader", "context switching"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "cloud", "docker", "kubernetes", "k8s", "ci/cd", "ci cd", "jenkins", "gitlab", "github actions", "terraform", "ansible", "devops", "sre", "serverless", "rds", "azure sql", "cloudwatch", "prometheus", "grafana", "logging", "monitoring", "openstack", "digitalocean", "heroku", "cloudformation", "pivotal cloud", "microservices", "load balancing", "infrastructure as code", "containerization", "vmware", "configuration management", "scaling", "orchestration", "cloud migration", "eks", "gke", "aks"],
    "Tools & Concepts": ["git", "api", "rest", "restful", "graphql", "oop", "object oriented", "data structures", "data structure", "algorithms", "algorithm", "machine learning", "networking", "security", "testing", "unit test", "integration test", "agile", "scrum", "jira", "pandas", "numpy", "celery", "kafka", "rabbitmq", "version control", "continuous integration", "design patterns", "design pattern", "recursion", "functional programming", "bit manipulation", "system design", "distributed systems", "ci/cd pipelines", "docker compose", "reactivity", "reactive programming", "performance optimization", "profiling", "debugging", "code review", "pair programming", "dependency injection", "refactoring", "event driven", "concurrency", "parallel processing", "http", "dns", "tcp/ip", "ssl", "tls", "authentication", "authorization", "encryption"],
    "Mobile Development": ["android", "ios", "swift", "kotlin", "react native", "flutter", "mobile app", "objective-c", "mobile development", "cordova", "xamarin", "mobile ui", "mobile ux", "hybrid app", "mobile optimization", "android studio", "xcode", "phonegap", "ionic", "cross platform", "appcelerator", "progressive web app"],
    "Data Science & Analytics": ["machine learning", "data analysis", "statistics", "r", "python", "data visualization", "matplotlib", "seaborn", "scikit-learn", "sklearn", "pandas", "numpy", "tensorflow", "keras", "pytorch", "deep learning", "data mining", "predictive analytics", "big data analytics", "data cleaning", "data wrangling", "statistical modeling", "spss", "stata", "business analytics", "data storytelling", "feature engineering", "dimensionality reduction", "model deployment", "ensemble methods", "clustering", "regression analysis", "time series analysis", "natural language processing", "nlp"],
    "UI/UX & Design": ["ui", "ux", "user interface", "user experience", "photoshop", "illustrator", "sketch", "figma", "adobe xd", "design", "prototyping", "wireframing", "responsive design", "graphic design", "visual design", "interaction design", "user research", "usability testing", "design thinking", "animation", "branding", "typography", "color theory", "layout", "information architecture", "motion design"],
    "Networking & Security": ["network", "networking", "cybersecurity", "firewall", "encryption", "ssl", "tls", "https", "penetration testing", "network security", "information security", "vulnerability", "incident response", "ethical hacking", "ids", "ips", "siem", "vpn", "wireshark", "security protocols", "malware analysis", "risk assessment", "access control", "network monitoring", "security audit", "intrusion detection", "ddos protection", "zero trust", "data loss prevention", "phishing prevention", "tcp/ip", "http", "dns", "dhcp", "routing", "switching", "lan", "wan", "osi model"],
    "Project Management & Collaboration": ["project management", "scrum", "kanban", "jira", "trello", "asana", "confluence", "slack", "agile", "stakeholder", "timeline", "risk management", "resource allocation", "gantt chart", "critical path", "project planning", "milestones", "budgeting", "communication", "team collaboration", "leadership", "project lifecycle", "sprint planning", "retrospective", "project charter", "deliverables"],
    "Game Development": ["unity", "unreal engine", "game development", "c#", "c++", "game design", "3d modeling", "animation", "virtual reality", "augmented reality", "gamedev", "game physics", "level design", "sprite", "game engine", "multiplayer", "shader", "pixel art", "game mechanics", "scripting", "sound design", "asset creation"],
    "Business Intelligence": ["bi", "tableau", "power bi", "qlik", "data warehousing", "reporting", "analytics", "data mining", "dashboards", "etl", "data integration", "data modeling", "olap", "ad hoc reporting", "self-service analytics", "data visualization", "kpi", "scorecards"],
    "Big Data": ["hadoop", "spark", "big data", "kafka", "flink", "hive", "mapreduce", "distributed computing", "nosql", "data lake", "storm", "cassandra", "data streaming", "data processing", "real-time analytics", "batch processing", "impala", "apache beam", "data ingestion", "spark sql"],
    "Testing & QA": ["unit testing", "integration testing", "selenium", "cucumber", "automation", "qa", "quality assurance", "performance testing", "load testing", "test automation", "bug tracking", "testrail", "regression testing", "smoke testing", "functional testing", "acceptance testing", "uat", "manual testing", "test plan", "test case", "usability testing", "security testing", "continuous testing", "pytest", "junit", "testng", "jest", "mocha", "cypress"],
    "IoT & Embedded Systems": ["iot", "embedded", "arduino", "raspberry pi", "firmware", "iot protocols", "sensor", "actuator", "real-time operating system", "rtos", "esp32", "embedded linux", "microcontroller", "bluetooth", "zigbee", "lte", "mcu", "pcb design", "embedded c", "robotics", "edge computing", "wearable technology", "mqtt"]
}

def extract_skills_from_text(text):
    """Extracts potential technical skills by matching predefined keywords."""
    found_skills = set()
    text_lower = " " + text.lower() + " "
    for category, keywords in SKILL_CATEGORIES.items():
        for keyword in keywords:
             try:
                 pattern_keyword = keyword.replace('+', r'\+').replace('#', r'\#').replace('.', r'\.')
                 pattern_keyword = pattern_keyword.replace(' ', r'[\s\-]+')
                 pattern = r'(?<!\w)' + pattern_keyword + r'(?!\w)'
                 if re.search(pattern, text_lower):
                     found_skills.add(keyword)
             except re.error as regex_err:
                 logger.warning(f"Regex error searching for keyword '{keyword}': {regex_err}")
                 continue
    final_skills = set()
    sorted_skills = sorted(list(found_skills), key=len, reverse=True)
    for skill in sorted_skills:
        is_substring = any(skill != longer and (f" {skill} " in f" {longer} " or longer.startswith(skill + " ") or longer.endswith(" " + skill)) for longer in final_skills)
        if not is_substring:
            final_skills.add(skill)
    logger.info(f"Extracted {len(final_skills)} potential skills: {list(final_skills)[:20]}...")
    return final_skills

# --- Query and Topic Generation ---
# (generate_search_queries and get_focus_topics remain the same as the previous refined version)
def generate_search_queries(resume_text, jd_text, num_queries=NUM_QUESTIONS + 1):
    """Generates search queries based on resume and JD keywords using predefined lists."""
    logger.info("Extracting potential skills from Resume...")
    resume_skills = extract_skills_from_text(resume_text)
    if not resume_skills: logger.warning("No predefined skills found in resume.")
    logger.info("Extracting potential skills from Job Description...")
    jd_skills = extract_skills_from_text(jd_text)
    if not jd_skills: logger.warning("No predefined skills found in job description.")
    if not resume_skills and not jd_skills:
        logger.error("Skill extraction failed for both resume and JD. Using default queries.")
        return ["Explain common data structures", "Describe REST API principles", "Explain Operating System concepts", "Describe database normalization", "What is version control with Git", "Explain SQL JOIN types"][:num_queries]
    intersection = sorted(list(resume_skills.intersection(jd_skills)), key=len, reverse=True)
    jd_only = sorted(list(jd_skills - resume_skills), key=len, reverse=True)
    resume_only = sorted(list(resume_skills - jd_skills), key=len, reverse=True)
    generic_terms = {"database", "concepts", "basic", "understanding", "tools", "skills", "programming", "technology", "systems", "system", "data", "web", "app", "experience", "knowledge", "design", "development", "management", "analytics", "administration"}
    intersection = [kw for kw in intersection if kw not in generic_terms and len(kw) > 2]
    jd_only = [kw for kw in jd_only if kw not in generic_terms and len(kw) > 2]
    resume_only = [kw for kw in resume_only if kw not in generic_terms and len(kw) > 2]
    queries = []
    db_keywords = set(SKILL_CATEGORIES["Databases"] + SKILL_CATEGORIES["Big Data"] + SKILL_CATEGORIES["Business Intelligence"])
    cloud_keywords = set(SKILL_CATEGORIES["Cloud & DevOps"])
    security_keywords = set(SKILL_CATEGORIES["Networking & Security"])
    queries.extend([f"Explain the concept and importance of {topic}" for topic in intersection[:max(2, num_queries // 3)]])
    queries.extend([
        f"Describe {topic} in database systems" if topic in db_keywords else f"Describe the application of {topic} in software engineering"
        for topic in jd_only[:max(2, num_queries // 3)]
    ])
    jd_text_lower = jd_text.lower()
    if len(queries) < num_queries and any(kw in jd_skills for kw in security_keywords):
         if not any("security" in q.lower() for q in queries):
             queries.append("Explain common database security principles")
    if len(queries) < num_queries and any(kw in jd_skills for kw in cloud_keywords):
         if not any("cloud" in q.lower() for q in queries):
             queries.append("Describe basic concepts of cloud database services")
    needed = num_queries - len(queries)
    if needed > 0:
        added_resume = 0
        for topic in resume_only:
            topic_base = topic.split()[0]
            if added_resume < needed and not any(topic_base in q.lower() for q in queries):
                 queries.append(f"Explain fundamental concepts of {topic}")
                 added_resume += 1
    needed = num_queries - len(queries)
    if needed > 0:
        general_defaults = ["data structures and algorithms", "object oriented programming principles", "database normalization forms", "operating system process management", "api design fundamentals", "version control branching strategies", "common system design patterns", "software testing types", "network protocols explained", "acid properties explained"]
        added_general = 0
        for topic in general_defaults:
            topic_base = topic.split()[0]
            if added_general < needed and not any(topic_base in q.lower() for q in queries):
                 queries.append(f"Explain {topic}")
                 added_general += 1
    final_queries = list(dict.fromkeys(queries))[:num_queries]
    if len(final_queries) < num_queries:
         logger.warning(f"Generated fewer than {num_queries} unique queries. Adding more defaults.")
         extra_defaults = ["Explain SQL JOIN types", "Describe database indexing", "Explain REST principles", "What is TCP/IP"]
         needed = num_queries - len(final_queries)
         added_extra = 0
         for topic in extra_defaults:
             if added_extra < needed and not any(topic.split()[0] in q.lower() for q in final_queries):
                  final_queries.append(topic)
                  added_extra += 1
    logger.info(f"Generated {len(final_queries)} search queries: {final_queries}")
    return final_queries

def get_focus_topics(resume_text, jd_text):
     logger.info("Identifying focus topics...")
     resume_skills = extract_skills_from_text(resume_text)
     jd_skills = extract_skills_from_text(jd_text)
     intersection = sorted(list(resume_skills.intersection(jd_skills)), key=len, reverse=True)
     key_jd_skills_patterns = ["sql", "postgresql", "mysql", "sql server", "oracle", "database administration", "dba", "backup", "recovery", "security", "query optimization", "indexing", "normalization", "cloud", "aws", "azure", "gcp", "docker", "kubernetes", "python", "java", "c++", "api", "rest", "data structures", "algorithms", "linux", "networking", "testing"]
     key_jd_skills = [s for s in jd_skills if s in key_jd_skills_patterns]
     focus_candidates = sorted(list(set(intersection + key_jd_skills)), key=len, reverse=True)
     generic_terms = {"database", "concepts", "basic", "understanding", "tools", "skills", "programming", "technology", "systems", "system", "data", "web", "app", "experience", "knowledge", "design", "development", "management", "analytics", "administration"}
     focus = [kw for kw in focus_candidates if kw not in generic_terms and len(kw) > 2]
     logger.info(f"Key focus topics identified for prompt: {focus[:15]}...")
     return focus