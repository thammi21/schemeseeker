# data/manual_schemes.py
"""
Manual scheme entries not present in the HuggingFace dataset.
Run this ONCE after ingestion to add missing schemes.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HF_CACHE_DIR    = BASE_DIR / "data" / "hf_cache"

MANUAL_SCHEMES = [
    {
        "question": "What is PM Kisan Samman Nidhi and who is eligible?",
        "answer"  : """PM Kisan Samman Nidhi is a Central Government scheme that provides
financial support of Rs 6000 per year to farmer families across India.
The amount is paid in three equal installments of Rs 2000 every four months
directly into the farmer's bank account via DBT.

ELIGIBILITY:
- All landholding farmer families with cultivable land in their name
- Both small and marginal farmers qualify
- Land must be in the farmer's or a family member's name
- Applicable across all states and UTs of India

WHO IS EXCLUDED:
- Former or current holders of constitutional posts
- Current or former Ministers, MPs, MLAs
- Government employees drawing monthly salary above Rs 10,000
- Income taxpayers (except farmers)
- Professionals like doctors, engineers, lawyers, CAs

HOW TO APPLY:
- Visit nearest Common Service Centre (CSC)
- Apply online at pmkisan.gov.in
- Documents needed: Aadhaar card, land records, bank account details

Official website: pmkisan.gov.in"""
    },
    {
        "question": "What is Ayushman Bharat PM-JAY and who can get it?",
        "answer"  : """Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY) is the
world's largest health insurance scheme providing coverage of Rs 5 lakh
per family per year for secondary and tertiary hospitalisation.

ELIGIBILITY:
- Families listed in SECC 2011 database
- Poor and vulnerable families in rural and urban areas
- No restriction on family size or age
- Covers pre-existing conditions from day one

BENEFITS:
- Rs 5 lakh health cover per family per year
- Cashless treatment at empanelled hospitals
- Covers 1,929 medical procedures
- Pre and post hospitalisation expenses covered

HOW TO APPLY:
- Check eligibility at pmjay.gov.in or mera.pmjay.gov.in
- Visit nearest empanelled hospital with Aadhaar card
- No registration needed — eligible families auto-enrolled

Official website: pmjay.gov.in"""
    },
    {
        "question": "What is PM Mudra Yojana and how can I get a loan?",
        "answer"  : """PM Mudra Yojana provides loans up to Rs 10 lakh to
non-corporate, non-farm small and micro enterprises.

THREE CATEGORIES:
1. Shishu: Loans up to Rs 50,000 (for starting businesses)
2. Kishor: Loans from Rs 50,000 to Rs 5 lakh (for growing businesses)
3. Tarun: Loans from Rs 5 lakh to Rs 10 lakh (for established businesses)

ELIGIBILITY:
- Any Indian citizen with a business plan
- Small manufacturers, shopkeepers, vendors, artisans
- No minimum income requirement
- No collateral required for Shishu and Kishor

HOW TO APPLY:
- Visit any public sector bank, private bank, MFI or NBFC
- Apply online at udyamimitra.in
- Documents: Aadhaar, PAN, business proof, bank statements

Official website: mudra.org.in"""
    },
    {
        "question": "What is Ayushman Bharat PM-JAY "
                    "and who can benefit from it?",
        "answer"  : """Ayushman Bharat Pradhan Mantri Jan Arogya
Yojana (PM-JAY) is India's flagship health insurance scheme
providing cashless treatment coverage of Rs 5 lakh per family
per year at empanelled hospitals.

WHO CAN BENEFIT:
- Families listed in SECC 2011 socio-economic database
- Poor and vulnerable families in both rural and urban areas
- No restriction on family size or age of members
- Covers pre-existing conditions from day one of enrollment

BENEFITS:
- Rs 5 lakh health cover per family per year
- Cashless and paperless treatment at 25,000+ empanelled hospitals
- Covers 1,929 different medical procedures
- Pre-hospitalisation and post-hospitalisation expenses included
- No premium to be paid by beneficiaries

HOW TO CHECK ELIGIBILITY AND APPLY:
- Check at mera.pmjay.gov.in or call 14555
- Visit any empanelled hospital with Aadhaar card
- Eligible families are automatically enrolled — no registration
- Download Ayushman card from pmjay.gov.in

Official website: pmjay.gov.in""",
    },
    {
        "question": "What crop insurance schemes exist "
                    "for farmers in Maharashtra?",
        "answer"  : """Several crop insurance schemes are available
for farmers including those in Maharashtra:

1. PRADHAN MANTRI FASAL BIMA YOJANA (PMFBY)
- Coverage: All food crops, oilseeds, annual commercial crops
- Premium: Maximum 2% for Kharif, 1.5% for Rabi crops
- Benefit: Full insured sum for crop loss due to natural calamities
- Apply through: Banks, CSCs, or pmfby.gov.in

2. RESTRUCTURED WEATHER BASED CROP INSURANCE (RWBCIS)
- Coverage: Weather-related crop losses
- Triggered by: Deviation in rainfall, temperature, humidity
- Premium: Same as PMFBY
- Apply through: Same channels as PMFBY

3. MAHARASHTRA SPECIFIC — GOPINATH MUNDE SHETKARI APGHAT VIMA YOJANA
- Coverage: Accidental insurance for farmers
- Benefit: Rs 2 lakh on accidental death, partial disability covered
- Eligibility: All farmers aged 10-75 years in Maharashtra
- Automatic: No separate application needed

Official website: pmfby.gov.in
Maharashtra agriculture dept: mahaagri.gov.in""",
    },
]


def add_manual_schemes():
    print("Loading embedding model...")
    emb = HuggingFaceEmbeddings(
        model_name    = "paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
        cache_folder  = str(HF_CACHE_DIR / "embeddings"),
    )

    vs = Chroma(
        collection_name    = "schemeseeker",
        embedding_function = emb,
        persist_directory  = str(VECTORSTORE_DIR),
    )

    print(f"Adding {len(MANUAL_SCHEMES)} manual schemes...")

    docs = []
    for scheme in MANUAL_SCHEMES:
        content = (
            f"QUESTION: {scheme['question']}\n\n"
            f"ANSWER:\n{scheme['answer']}"
        )
        docs.append(Document(
            page_content = content,
            metadata     = {
                "source"  : "manual_curation",
                "question": scheme["question"],
            }
        ))

    vs.add_documents(docs)
    print(f"Added {len(docs)} schemes to vectorstore")

    # Verify
    results = vs.similarity_search("PM Kisan Samman Nidhi", k=2)
    print(f"\nPM Kisan now found: {len(results)} chunks")
    if results:
        print(f"Preview: {results[0].page_content[:200]}")

    print("\nDone — manual schemes added successfully")


if __name__ == "__main__":
    add_manual_schemes()