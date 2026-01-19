"""
Job Matching System
Matches student profiles with company requirements
Returns top 5 company recommendations with fit scores
"""


import json
import pandas as pd


def load_company_database(json_path='src/company_database.json'):
    """Load company database from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['companies']


def calculate_match_score(student_profile, company):
    """
    Calculate match score between student and company
    Returns score out of 100
    """
    score = 0
    max_score = 100
    
    # 1. CGPA Check (30 points)
    if student_profile['cgpa'] >= company['min_cgpa']:
        score += 30
        # Bonus for exceeding requirement
        if student_profile['cgpa'] >= company['min_cgpa'] + 1:
            score += 5
    elif student_profile['cgpa'] >= company['min_cgpa'] - 0.5:
        # Partial credit if close
        score += 15
    
    # 2. 10th Marks Check (10 points)
    if student_profile['tenth_marks'] >= company['min_tenth']:
        score += 10
    elif student_profile['tenth_marks'] >= company['min_tenth'] - 5:
        score += 5
    
    # 3. 12th Marks Check (10 points)
    if student_profile['twelfth_marks'] >= company['min_twelfth']:
        score += 10
    elif student_profile['twelfth_marks'] >= company['min_twelfth'] - 5:
        score += 5
    
    # 4. Skills Matching (40 points)
    student_skills = [s.lower().strip() for s in student_profile.get('skills', [])]
    required_skills = [s.lower().strip() for s in company['skills_required']]
    
    if len(required_skills) > 0:
        matched_skills = []
        for req_skill in required_skills:
            # Check exact match or partial match
            for student_skill in student_skills:
                if req_skill in student_skill or student_skill in req_skill:
                    matched_skills.append(req_skill)
                    break
        
        skill_match_ratio = len(matched_skills) / len(required_skills)
        skill_score = skill_match_ratio * 40
        score += skill_score
    else:
        score += 20  # If no specific skills required
    
    # 5. Experience Bonus (10 points)
    experience_score = 0
    
    if student_profile.get('internships', 0) > 0:
        experience_score += 3
    
    if student_profile.get('projects', 0) > 0:
        experience_score += 3
    
    if student_profile.get('training', 0) > 0:
        experience_score += 2
    
    if student_profile.get('technical_course', 0) > 0:
        experience_score += 2
    
    score += min(experience_score, 10)  # Cap at 10
    
    return min(score, max_score)  # Ensure max 100


def get_top_matches(student_profile, predicted_category=None, top_n=5):
    """
    Get top N company matches for student
    Filters by predicted category if provided
    """
    
    companies = load_company_database()
    
    # Filter by predicted category if provided
    if predicted_category:
        companies = [c for c in companies if c['package_category'] == predicted_category]
    
    # Calculate scores for all companies
    matches = []
    for company in companies:
        match_score = calculate_match_score(student_profile, company)
        
        # Get matched skills
        student_skills = [s.lower().strip() for s in student_profile.get('skills', [])]
        required_skills = [s.lower().strip() for s in company['skills_required']]
        matched_skills = []
        
        for req_skill in required_skills:
            for student_skill in student_skills:
                if req_skill in student_skill or student_skill in req_skill:
                    matched_skills.append(req_skill.title())
                    break
        
        matches.append({
            'company': company['name'],
            'role': company['role'],
            'location': company['location'],
            'package': f"₹{company['package_min']}-{company['package_max']} LPA",
            'package_category': company['package_category'],
            'match_score': round(match_score, 1),
            'skills_required': company['skills_required'],
            'skills_matched': matched_skills,
            'skills_gap': [s for s in company['skills_required'] if s.lower() not in [m.lower() for m in matched_skills]],
            'cgpa_required': company['min_cgpa'],
            'meets_cgpa': student_profile['cgpa'] >= company['min_cgpa'],
            'focus': company['focus']
        })
    
    # Sort by match score
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    return matches[:top_n]


def display_matches(matches):
    """Display matches in formatted way"""
    
    if not matches:
        print("❌ No matching companies found!")
        return
    
    print("\n" + "="*70)
    print("🎯 TOP COMPANY RECOMMENDATIONS FOR YOU")
    print("="*70)
    
    for i, match in enumerate(matches, 1):
        print(f"\n{'='*70}")
        print(f"#{i} - {match['company']} | Match Score: {match['match_score']}%")
        print(f"{'='*70}")
        
        print(f"📍 Location: {match['location']}")
        print(f"💼 Role: {match['role']}")
        print(f"💰 Package: {match['package']} ({match['package_category']})")
        print(f"🎯 CGPA Required: {match['cgpa_required']} | You: {'✅ MEET' if match['meets_cgpa'] else '❌ DON\'T MEET'}")
        print(f"📊 Focus: {match['focus'].replace('_', ' ').title()}")
        
        print(f"\n✅ Skills You Have ({len(match['skills_matched'])}):")
        if match['skills_matched']:
            print(f"   {', '.join(match['skills_matched'])}")
        else:
            print("   None matched exactly")
        
        print(f"\n⚠️ Skills You Need ({len(match['skills_gap'])}):")
        if match['skills_gap']:
            print(f"   {', '.join(match['skills_gap'])}")
        else:
            print("   ✅ You have all required skills!")
    
    print("\n" + "="*70)


# Test function
if __name__ == "__main__":
    # Example student profile
    test_student = {
        'cgpa': 7.5,
        'tenth_marks': 75,
        'twelfth_marks': 75,
        'skills': ['Python', 'Java', 'DSA', 'SQL', 'React'],
        'internships': 1,
        'projects': 2,
        'training': 1,
        'technical_course': 1
    }
    
    print("🎓 TESTING JOB MATCHER")
    print("="*70)
    print(f"Student Profile: CGPA {test_student['cgpa']}")
    print(f"Skills: {', '.join(test_student['skills'])}")
    
    # Get all matches
    matches = get_top_matches(test_student, top_n=5)
    display_matches(matches)
    
    # Get matches for specific category
    print("\n\n🎯 PREMIUM COMPANIES ONLY:")
    print("="*70)
    premium_matches = get_top_matches(test_student, predicted_category='Premium', top_n=3)
    display_matches(premium_matches)
