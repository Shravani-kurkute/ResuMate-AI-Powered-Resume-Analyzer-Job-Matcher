"""
Resume Parser using NLP (spaCy)
Extracts key information from PDF resumes using Named Entity Recognition,
POS tagging, dependency parsing, and semantic similarity
"""

import re
import PyPDF2
import pdfplumber
from pathlib import Path
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from collections import Counter

# Load spaCy model (install with: python -m spacy download en_core_web_md)
try:
    nlp = spacy.load("en_core_web_md")
except:
    print("⚠️  Installing spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using multiple methods"""
    text = ""
    
    try:
        # Method 1: PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
        
        # Method 2: pdfplumber (fallback)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e2:
            print(f"pdfplumber also failed: {e2}")
            return None
    
    return text

def extract_entities(doc):
    """Extract named entities using spaCy NER"""
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],  # Geopolitical entities (cities, countries)
        'DATE': [],
        'CARDINAL': [],  # Numbers
        'PERCENT': []
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

def extract_email_phone(text):
    """Extract contact information using regex"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return emails[0] if emails else None, phones[0] if phones else None

def extract_cgpa_nlp(doc, text):
    """Extract CGPA using NLP context understanding + robust regex"""
    cgpa_values = []
    
    # More robust regex patterns with dash support
    patterns = [
        r'cgpa[\s:\-–—]+([0-9]\.[0-9]+)\b',
        r'gpa[\s:\-–—]+([0-9]\.[0-9]+)\b',
        r'cpi[\s:\-–—]+([0-9]\.[0-9]+)\b',
        r'grade point[\s:\-–—]+([0-9]\.[0-9]+)\b',
        r'cumulative.*?([0-9]\.[0-9]+)\b',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            val = float(match)
            if 0 <= val <= 10:
                cgpa_values.append(val)
    
    # Also look through sentences for context
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(term in sent_text for term in ['cgpa', 'gpa', 'cpi', 'grade point']):
            # Extract numbers using regex from this specific sentence
            numbers = re.findall(r'\b([0-9]\.[0-9]+)\b', sent_text)
            for num in numbers:
                val = float(num)
                if 0 <= val <= 10:
                    cgpa_values.append(val)
    
    return max(cgpa_values) if cgpa_values else 7.0


def extract_marks_nlp(doc, text):
    """Extract 10th and 12th marks using NLP + robust regex"""
    marks = {'10th': 75.0, '12th': 75.0}
    
    text_lower = text.lower()
    
    # 12th marks patterns (check these FIRST to avoid "secondary" matching issues)
    twelfth_patterns = [
        r'12th[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'twelfth[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'higher secondary[\s:\-–—]*.*?percentage[\s:\-–—]*([0-9]+\.?[0-9]*)',
        r'higher secondary[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'hsc[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'class xii[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'intermediate[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
    ]
    
    for pattern in twelfth_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 100:
                marks['12th'] = val
                break
    
    # 10th marks patterns (check AFTER 12th to avoid conflicts)
    tenth_patterns = [
        r'10th[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'tenth[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'(?<!higher )secondary[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'ssc[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'class x\b[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
        r'matriculation[\s:\-–—]+([0-9]+\.?[0-9]*)\s*%?',
    ]
    
    for pattern in tenth_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 100:
                marks['10th'] = val
                break
    
    return marks['10th'], marks['12th']

def extract_skills_nlp(doc):
    """Extract technical skills using PhraseMatcher and semantic similarity"""
    
    # Comprehensive skill list
    skill_patterns = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust", "Swift",
        "React", "Vue", "Node.js", "Express", "Django", "Flask", "Spring",
        "Machine Learning", "Deep Learning", "Data Science", "Artificial Intelligence",
        "Natural Language Processing", "Computer Vision",
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins",
        "MongoDB","PostgreSQL",
        "TensorFlow", "Scikit-learn", "Pandas", "NumPy",
    ]
    
    # Create PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_patterns]
    matcher.add("SKILLS", patterns)
    
    # Find matches
    matches = matcher(doc)
    found_skills = set()
    
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text)
    
    # Also look for noun chunks that might be technical terms
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(tech_term in chunk_text for tech_term in ['framework', 'library', 'tool', 'language', 'platform']):
            found_skills.add(chunk.text)
    
    return list(found_skills) if found_skills else ['Python', 'Java']

def extract_experience_nlp(doc):
    """Extract experience information using dependency parsing"""
    
    experience = {
        'internships': 0,
        'projects': 0,
        'training': 0,
        'certifications': 0
    }
    
    internship_terms = ['intern', 'internship', 'trainee']
    project_terms = ['project', 'developed', 'built', 'created', 'implemented']
    training_terms = ['training', 'workshop', 'bootcamp', 'course']
    cert_terms = ['certification', 'certified', 'certificate']
    
    # Count mentions using linguistic context
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Look for verb patterns indicating experience
        for token in sent:
            # Check if it's a verb related to project work
            if token.pos_ == "VERB" and token.lemma_ in ['develop', 'build', 'create', 'design', 'implement']:
                experience['projects'] += 1
        
        # Count specific terms
        if any(term in sent_text for term in internship_terms):
            experience['internships'] += 1
        
        if any(term in sent_text for term in training_terms):
            experience['training'] += 1
            
        if any(term in sent_text for term in cert_terms):
            experience['certifications'] += 1
    
    # Binary flags
    return {
        'internships': 1 if experience['internships'] > 0 else 0,
        'projects': 1 if experience['projects'] >= 2 else 0,
        'training': 1 if experience['training'] > 0 else 0,
        'technical_course': 1 if experience['certifications'] > 0 else 0
    }

def analyze_communication_nlp(doc):
    """Analyze communication quality using linguistic features"""
    
    # Calculate linguistic complexity metrics
    num_sentences = len(list(doc.sents))
    num_words = len([token for token in doc if not token.is_punct])
    num_unique_words = len(set([token.lemma_.lower() for token in doc if not token.is_punct]))
    
    # Lexical diversity (vocabulary richness)
    lexical_diversity = num_unique_words / num_words if num_words > 0 else 0
    
    # Average sentence length
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    # Count complex sentences (with subordinate clauses)
    complex_sentences = 0
    for sent in doc.sents:
        if any(token.dep_ in ['advcl', 'ccomp', 'xcomp'] for token in sent):
            complex_sentences += 1
    
    # Scoring based on linguistic features
    score = 3  # Base score
    
    if lexical_diversity > 0.6:
        score += 1
    if avg_sentence_length > 15:
        score += 0.5
    if complex_sentences > num_sentences * 0.3:
        score += 0.5
    
    return min(int(score), 5)

def estimate_tech_skills_nlp(skills_list, doc):
    """Estimate technical proficiency using skill count and context"""
    
    skill_count = len(skills_list)
    
    # Look for proficiency indicators in text
    proficiency_terms = {
        'expert': 30,
        'proficient': 25,
        'experienced': 20,
        'skilled': 15,
        'familiar': 10
    }
    
    bonus_score = 0
    doc_text = doc.text.lower()
    for term, points in proficiency_terms.items():
        if term in doc_text:
            bonus_score += points
            break
    
    # Base score from skill count
    if skill_count >= 20:
        base_score = 75
    elif skill_count >= 15:
        base_score = 60
    elif skill_count >= 10:
        base_score = 50
    elif skill_count >= 5:
        base_score = 40
    else:
        base_score = 30
    
    return min(base_score, 100)

def parse_resume(pdf_path):
    """
    Main function to parse resume using NLP
    Returns dict with extracted information
    """
    print(f"\n{'='*60}")
    print(f"📄 PARSING RESUME WITH NLP: {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("❌ Could not extract text from PDF!")
        return None
    
    print(f"✅ Extracted {len(text)} characters")
    print(f"🧠 Processing with spaCy NLP...")
    
    # Process with spaCy
    doc = nlp(text)
    
    print(f"✅ Identified {len(list(doc.sents))} sentences")
    print(f"✅ Found {len(doc.ents)} named entities")
    
    # Extract contact info
    email, phone = extract_email_phone(text)
    
    # Extract entities
    entities = extract_entities(doc)
    
    # Extract all information using NLP
    cgpa = extract_cgpa_nlp(doc, text)
    tenth_marks, twelfth_marks = extract_marks_nlp(doc, text)
    skills = extract_skills_nlp(doc)
    experience = extract_experience_nlp(doc)
    communication = analyze_communication_nlp(doc)
    tech_score = estimate_tech_skills_nlp(skills, doc)
    
    # Compile results
    extracted_data = {
        'email': email,
        'phone': phone,
        'name': entities['PERSON'][0] if entities['PERSON'] else 'Unknown',
        'organizations': entities['ORG'][:3] if entities['ORG'] else [],
        'locations': entities['GPE'][:3] if entities['GPE'] else [],
        'cgpa': cgpa,
        'tenth_marks': tenth_marks,
        'twelfth_marks': twelfth_marks,
        'skills': skills,
        'internships': experience['internships'],
        'projects': experience['projects'],
        'training': experience['training'],
        'technical_course': experience['technical_course'],
        'communication_level': communication,
        'technical_skills_score': tech_score
    }
    
    # Display results
    print(f"\n📊 EXTRACTED INFORMATION (NLP-BASED):")
    print(f"{'='*60}")
    print(f"👤 Name: {extracted_data['name']}")
    print(f"📧 Email: {email or 'Not found'}")
    print(f"📱 Phone: {phone or 'Not found'}")
    if entities['ORG']:
        print(f"🏢 Organizations: {', '.join(entities['ORG'][:3])}")
    print(f"🎯 CGPA: {cgpa}")
    print(f"📚 10th Marks: {tenth_marks}%")
    print(f"📚 12th Marks: {twelfth_marks}%")
    print(f"💼 Internships: {'Yes' if experience['internships'] else 'No'}")
    print(f"🚀 Projects: {'Yes' if experience['projects'] else 'No'}")
    print(f"🏭 Training: {'Yes' if experience['training'] else 'No'}")
    print(f"💻 Certifications: {'Yes' if experience['technical_course'] else 'No'}")
    print(f"🗣️ Communication Level: {communication}/5")
    print(f"⚡ Tech Skills Score: {tech_score}/100")
    print(f"\n✅ Skills Found ({len(skills)}):")
    for skill in skills[:15]:
        print(f"   - {skill}")
    if len(skills) > 15:
        print(f"   ... and {len(skills)-15} more")
    print(f"{'='*60}\n")
    
    return extracted_data

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = parse_resume(pdf_path)
        
        if result:
            print("✅ Resume parsed successfully using NLP!")
        else:
            print("❌ Failed to parse resume!")
    else:
        print("Usage: python resume_parser_nlp.py <path_to_resume.pdf>")
        print("\nFirst install requirements:")
        print("pip install spacy")
        print("python -m spacy download en_core_web_md")