# SchemeSeeker — RAG Evaluation Report

**Date:** 2026-09-07
**Total tests:** 20

## Overall Scores

| Metric | Score |
|--------|-------|
| Overall Score | 70.2% |
| English Queries | 78.7% |
| Hindi Queries | 36.5% |
| On-Topic Score | 66.9% |
| Fallback Accuracy | 100.0% |
| Unnecessary Fallbacks | 2 |

## Results by Category

| ID | Category | Language | Score | Fallback | Keywords |
|----|----------|----------|-------|----------|----------|
| agr_01 | agriculture | english | ✅ 1.00 | No | 4/4 |
| agr_02 | agriculture | english | ✅ 1.00 | No | 3/3 |
| agr_03 | agriculture | hindi | ✅ 0.64 | No | 2/2 |
| agr_04 | agriculture | english | ❌ 0.00 | Yes | 0/3 |
| hlth_01 | health | english | ✅ 0.73 | No | 1/3 |
| hlth_02 | health | hindi | ⚠️ 0.44 | No | 1/2 |
| hlth_03 | health | english | ✅ 1.00 | No | 3/3 |
| edu_01 | education | english | ✅ 0.82 | No | 4/4 |
| edu_02 | education | english | ✅ 1.00 | No | 3/3 |
| edu_03 | education | hindi | ⚠️ 0.38 | No | 1/2 |
| hous_01 | housing | english | ✅ 0.87 | No | 2/3 |
| hous_02 | housing | english | ✅ 0.73 | No | 1/3 |
| fin_01 | finance | english | ✅ 0.82 | No | 3/3 |
| fin_02 | finance | english | ✅ 0.73 | No | 1/3 |
| fin_03 | finance | hindi | ❌ 0.00 | Yes | 0/2 |
| wom_01 | women | english | ✅ 1.00 | No | 2/2 |
| wom_02 | women | english | ✅ 0.82 | No | 3/3 |
| oot_01 | off_topic | english | ✅ 1.00 | Yes | 0/0 |
| oot_02 | off_topic | english | ✅ 1.00 | Yes | 0/0 |
| full_01 | agriculture | english | ❌ 0.06 | No | 0/3 |

## Failed Tests

### agr_04 — What is PM Kisan Samman Nidhi?
- Score: 0.00
- Fell back: True
- Answer preview: I don't have specific information about this in my database. Please check myscheme.gov.in

### hlth_02 — महिलाओं के लिए स्वास्थ्य योजनाएं कौन सी हैं?
- Score: 0.44
- Fell back: False
- Answer preview: **Swasthya Sathi** (West Bengal)  
- **पात्रता:** उपलब्ध जानकारी में पात्रता का विवरण नहीं दिया गया है।  
- **लाभ:** प्रति परिवार ₹5 लाख तक वार्षिक कवरेज, एंपैनल्ड अस्पतालों में कैशलेस ट्रीटमेंट, पहले

### edu_03 — छात्रवृत्ति योजनाएं कौन सी हैं?
- Score: 0.38
- Fell back: False
- Answer preview: **1. स्कीम फॉर अवॉर्ड ऑफ स्कॉलरशिप्स टू स्कूल या कॉलेज‑गोइंग चिल्ड्रेन ऑफ एम्प्लॉइज़/इंडस्ट्रियल एम्प्लॉइज़ इन वैरियस एस्टैब्लिशमेंट्स**  
- **पात्रता**  
  - सरकारी/निजी संस्थानों में कार्यरत कर्मचार

### fin_03 — व्यापार के लिए ऋण योजनाएं कौन सी हैं?
- Score: 0.00
- Fell back: True
- Answer preview: मुझे इस बारे में जानकारी नहीं है। कृपया myscheme.gov.in देखें।

### full_01 — I am a 35 year old woman farmer in Maharashtra with 2 acres 
- Score: 0.06
- Fell back: False
- Answer preview: ERROR: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01k4t6h9rqf66bfqn9p9b6mswe` service tier `on_demand` on tokens per minute (TPM):
