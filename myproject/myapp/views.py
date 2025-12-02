from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.http import HttpResponse
from django.db import models
from django.views.decorators.csrf import csrf_exempt
from .models import ITEM
from django.conf import settings
import json
import numpy as np
import re

try:
    from google import genai
except Exception:
    genai = None

# Initialize Gemini client if available
_gemini_client = None
if genai is not None and getattr(settings, 'GEMINI_API_KEY', ''):
    try:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception:
        _gemini_client = None


def dp_average(ratings, a, b, epsilon):
    """
    Differentially private average using Laplace mechanism.
    """
    n = len(ratings)
    if n == 0:
        return 0.0, 0.0
    
    true_avg = np.mean(ratings)
    
    # sensitivity for average
    sensitivity = (b - a) / n
    
    # Laplace noise scale
    scale = sensitivity / epsilon
    
    # Adding Laplace noise to the true average
    noisy_avg = true_avg + np.random.laplace(0, scale)
    
    return noisy_avg, true_avg


def dp_difficulty_average(difficulty, a, b, epsilon):
    """
    Differentially private average using Laplace mechanism.
    Returns only the noisy average.
    """
    n = len(difficulty)
    true_avg = np.mean(difficulty)
   
    sensitivity = (b - a) / n
    scale = sensitivity / epsilon
    noisy_avg = true_avg + np.random.laplace(0, scale)
    return noisy_avg, true_avg


def dp_count(count, epsilon):
    """
    Differentially private count using Laplace mechanism.
    Sensitivity for a count query is 1.0.
    """
    # Sensitivity for a count query is always 1.0
    sensitivity = 1.0
    
    scale = sensitivity / epsilon
    noisy_count = count + np.random.laplace(0, scale)
    
    return noisy_count

def dp_helpful_average(helpful, a, b, epsilon):
    """
    Differentially private average using Laplace mechanism.
    Returns only the noisy average.
    """
    n = len(helpful)
    if n == 0:
        return 0.0, 0.0

    true_avg = np.mean(helpful)

    # sensitivity for average
    sensitivity = (b - a) / n

    # Laplace noise scale
    scale = sensitivity / epsilon

    # Adding Laplace noise to the true average
    noisy_avg = true_avg + np.random.laplace(0, scale)

    # Clamp to keep value within valid bounds (non-negative)
    noisy_avg = max(a, min(b, noisy_avg))

    return noisy_avg, true_avg



def detect_and_remove_personal_info(text: str) -> tuple[bool, str]:
    """
    Detect personal information using regex patterns and remove it.
    Returns (has_personal_info: bool, cleaned_text: str)
    """
    if not text:
        return False, text
    
    original_text = text
    has_personal_info = False
    
    # Pattern for email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, text):
        has_personal_info = True
        text = re.sub(email_pattern, '[email removed]', text)
    
    # Pattern for phone numbers (various formats)
    # More specific patterns to avoid false positives like years
    phone_patterns = [
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',  # US format with separators: 123-456-7890
        r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',  # 123.456.7890 or 123 456 7890
        r'\b\+?\d{1,3}[-.\s]\d{1,4}[-.\s]\d{1,4}[-.\s]\d{1,9}\b',  # International with separators
        # 10 consecutive digits but not at start of line (to avoid matching years in dates)
        r'(?<!\d)\d{10}(?!\d)',  # 10 digits not preceded or followed by digits
    ]
    for pattern in phone_patterns:
        if re.search(pattern, text):
            has_personal_info = True
            text = re.sub(pattern, '[phone number removed]', text)
    
    # Pattern for common name indicators (e.g., "My name is John", "I'm Sarah")
    name_patterns = [
        r'\b(?:my name is|i\'?m|i am|this is|call me|named|i go by)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
        r'\b(?:signed|from|yours)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',  # Email signatures
        # Pattern for standalone capitalized names (likely to be names if not common words)
        r'\b(?:hi|hello|hey|dear)\s+[A-Z][a-z]{2,}\b',  # "Hi John" or "Hello Sarah"
    ]
    for pattern in name_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            has_personal_info = True
            text = re.sub(pattern, '[name removed]', text, flags=re.IGNORECASE)
    
    
    # Pattern for ID numbers with common prefixes
    id_prefix_patterns = [
        r'\b(?:id|student id|studentid|student number|student#|sid|uid|user id|userid)\s*:?\s*[A-Z0-9]{4,}\b',
        r'\b(?:id|student id|studentid|student number|student#|sid|uid|user id|userid)\s*:?\s*\d{6,}\b',
    ]
    for pattern in id_prefix_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            has_personal_info = True
            text = re.sub(pattern, '[ID number removed]', text, flags=re.IGNORECASE)
    
    # Pattern for standalone ID numbers (6-12 digits, likely to be IDs)
    # But avoid matching years, phone numbers, or other common numbers
    standalone_id_pattern = r'\b(?:id|#)\s*\d{6,12}\b'
    if re.search(standalone_id_pattern, text, re.IGNORECASE):
        has_personal_info = True
        text = re.sub(standalone_id_pattern, '[ID number removed]', text, flags=re.IGNORECASE)
    
    # Pattern for alphanumeric ID numbers (common in student IDs, employee IDs)
    alphanumeric_id_pattern = r'\b[A-Z]{1,3}\d{4,10}\b|\b\d{4,10}[A-Z]{1,3}\b'
    if re.search(alphanumeric_id_pattern, text):
        has_personal_info = True
        text = re.sub(alphanumeric_id_pattern, '[ID number removed]', text)
    
    
    return has_personal_info, text


def make_review_private(review_text: str) -> str:
    """Use Gemini to anonymize the review text. If unavailable, return original."""
    if not review_text:
        return review_text
    
    # First, use regex to remove obvious personal info
    has_personal, cleaned = detect_and_remove_personal_info(review_text)
    if has_personal:
        review_text = cleaned
    
    if _gemini_client is None:
        return review_text

    prompt = f"""
You are an AI assistant ensuring differential privacy in student reviews.

Here is a student's review of a professor:
---
{review_text}
---

If this review contains personal or identifying information (like the student's name, schedule, project topic, group name, nationality, unique incidents, or specific grades),
rewrite it in a way that keeps the general opinion but removes or generalizes any identifying details, Also if it has email or phone number or name remove them.

If it is already anonymous and safe, just return the same text.

Return only the cleaned review, nothing else.
"""

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        cleaned = response.text.strip() if getattr(response, 'text', None) else review_text
        return cleaned or review_text
    except Exception:
        return review_text


@csrf_exempt
def check_privacy_risk(request):
    """Check privacy risk level and return rephrased text if high risk."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        review_text = data.get('review_text', '').strip()
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'error': 'Invalid request data'}, status=400)
    
    if not review_text:
        return JsonResponse({
            'risk_level': 'low',
            'original_text': '',
            'rephrased_text': ''
        })
    
    # First, check for obvious personal information using regex
    has_personal_info, regex_cleaned = detect_and_remove_personal_info(review_text)
    
    # If Gemini is not available, use regex-based detection
    if _gemini_client is None:
        if has_personal_info:
            return JsonResponse({
                'risk_level': 'high',
                'original_text': review_text,
                'rephrased_text': regex_cleaned,
                'error': 'AI service unavailable, using pattern-based detection'
            })
        else:
            return JsonResponse({
                'risk_level': 'low',
                'original_text': review_text,
                'rephrased_text': review_text,
                'error': 'AI service unavailable'
            })
    
    # Use the regex-cleaned version for AI analysis if personal info was found
    text_for_analysis = regex_cleaned if has_personal_info else review_text
    
    prompt = f"""
You are an AI that ensures differential privacy in student feedback.

The following text is a student's review of a professor:

---
{text_for_analysis}
---

Your task is to analyze this review for personal or identifying information such as:
- Student's name, email addresses, phone numbers
- Schedule, specific dates/times, class times
- Project topics, group names, team member names
- Nationality, ethnicity, or other personal identifiers
- Unique incidents that could identify the student
- Specific grades, scores, or exam results
- Student ID numbers or other identifiers
- Personal schedules or specific meeting times

CRITICAL INSTRUCTIONS:
1. If this review contains ANY identifying information (names, emails, phone numbers, specific dates, unique incidents, etc.), you MUST:
   - Set risk_level to "high"
   - Provide a rephrased version that COMPLETELY REMOVES all identifying information
   - Keep the general opinion and sentiment but remove ALL personal details
   - Replace names with generic terms like "the student" or "a classmate"
   - Remove or generalize specific dates, times, and unique incidents
   - Remove email addresses, phone numbers, and other contact information

2. If the review is already anonymous and safe (no identifying information), set risk_level to "low" and return the original text as rephrased_text.

3. You MUST return ONLY valid JSON in this exact format (no markdown, no code blocks, no additional text):
{{
    "risk_level": "high",
    "rephrased_text": "the cleaned version with all personal information removed"
}}

or

{{
    "risk_level": "low",
    "rephrased_text": "the original text here"
}}
"""
    
    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip() if getattr(response, 'text', None) else ''
        
        # Try to extract JSON from the response
        # Sometimes the model wraps JSON in markdown code blocks
        cleaned_raw = raw.strip()
        if cleaned_raw.startswith('```'):
            # Remove markdown code block markers
            lines = cleaned_raw.split('\n')
            if len(lines) > 2:
                cleaned_raw = '\n'.join(lines[1:-1]).strip()
        elif cleaned_raw.startswith('```json'):
            lines = cleaned_raw.split('\n')
            if len(lines) > 2:
                cleaned_raw = '\n'.join(lines[1:-1]).strip()
        
        # Try to find JSON object in the response
        json_start = cleaned_raw.find('{')
        json_end = cleaned_raw.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned_raw = cleaned_raw[json_start:json_end + 1]
        
        # Try to parse JSON
        try:
            result = json.loads(cleaned_raw)
            risk_level = result.get('risk_level', 'unknown').lower()
            rephrased_text = result.get('rephrased_text', review_text)
            
            # Validate risk_level
            if risk_level not in ['high', 'low']:
                risk_level = 'unknown'
            
            # Ensure rephrased_text is not empty
            if not rephrased_text or not rephrased_text.strip():
                rephrased_text = review_text
            
            # Always apply regex cleaning to AI output to catch anything AI might have missed
            _, final_cleaned = detect_and_remove_personal_info(rephrased_text)
            
            # If regex found personal info (either initially or in AI output), set risk to high
            if has_personal_info or final_cleaned != rephrased_text:
                risk_level = 'high'
                rephrased_text = final_cleaned
            # If regex didn't find anything initially, but AI says high risk, trust AI
            elif risk_level == 'high':
                # Still apply regex as a safety check
                rephrased_text = final_cleaned if final_cleaned != rephrased_text else rephrased_text
            
            return JsonResponse({
                'risk_level': risk_level,
                'original_text': review_text,
                'rephrased_text': rephrased_text
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, use regex detection as fallback
            if has_personal_info:
                return JsonResponse({
                    'risk_level': 'high',
                    'original_text': review_text,
                    'rephrased_text': regex_cleaned
                })
            
            # Try to determine risk from response
            # If the response is different from original, likely high risk
            cleaned_raw = cleaned_raw.strip()
            if cleaned_raw and cleaned_raw.lower() != review_text.lower() and len(cleaned_raw) > 10:
                # Apply regex cleaning to the AI response as well
                _, ai_cleaned = detect_and_remove_personal_info(cleaned_raw)
                return JsonResponse({
                    'risk_level': 'high',
                    'original_text': review_text,
                    'rephrased_text': ai_cleaned if ai_cleaned != cleaned_raw else cleaned_raw
                })
            else:
                return JsonResponse({
                    'risk_level': 'low',
                    'original_text': review_text,
                    'rephrased_text': review_text
                })
    except Exception as e:
        # On error, use regex detection as fallback
        if has_personal_info:
            return JsonResponse({
                'risk_level': 'high',
                'original_text': review_text,
                'rephrased_text': regex_cleaned,
                'error': f'AI error: {str(e)}, using pattern-based detection'
            })
        return JsonResponse({
            'risk_level': 'unknown',
            'original_text': review_text,
            'rephrased_text': review_text,
            'error': str(e)
        })

# Create your views here.
def home(request):
    # Get some statistics for the homepage
    total_professors = ITEM.objects.values_list('professor_name', flat=True).distinct().count()
    total_schools = ITEM.objects.values_list('school_name', flat=True).distinct().count()
    total_reviews = ITEM.objects.count()
    
    # Handle search functionality
    if request.method == 'POST':
        search_query = request.POST.get('search', '').strip()
        if search_query:
            # Search for professors by name
            matching_professors = ITEM.objects.filter(
                professor_name__icontains=search_query
            ).values_list('professor_name', flat=True).distinct()
            
            if matching_professors.exists():
                # If we find exact matches, redirect to the first professor's profile
                first_professor = matching_professors.first()
                return redirect('professor_profile', professor_name=first_professor)
            else:
                # If no exact matches, redirect to browse page with search results
                return redirect('showitems')
    
    context = {
        'total_professors': total_professors,
        'total_schools': total_schools,
        'total_reviews': total_reviews,
    }
    return render(request, 'home.html', context) 
    
    

# def register_view(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         confirm_password = request.POST.get('confirm_password')

#         if password == confirm_password:
#             try:
#                 user = User.objects.create_user(username=username, password=password)
#                 user.save()
#                 login(request, user)
#                 return redirect('home')
#             except:
#                 messages.error(request, "Username already exists.")
#         else:
#             messages.error(request, "Passwords do not match.")
#     return render(request, "registration/register.html", {'messages': "Register to Rate My Professor!"})
    
def showitems(request):
    # Get unique school names and professor names from the database
    schools = ITEM.objects.values_list('school_name', flat=True).distinct().order_by('school_name')
    professors = ITEM.objects.values_list('professor_name', flat=True).distinct().order_by('professor_name')
    
    selected_school = None
    selected_professor = None
    professor_details = None
    filtered_professors = professors  # Default to all professors
    
    if request.method == 'POST':
        selected_school = request.POST.get('school')
        selected_professor = request.POST.get('professor')
        
        # Filter professors by selected school
        if selected_school:
            filtered_professors = ITEM.objects.filter(school_name=selected_school).values_list('professor_name', flat=True).distinct().order_by('professor_name')
        
        # Get professor details if professor is selected
        if selected_professor:
            professor_details = ITEM.objects.filter(professor_name=selected_professor)
            if selected_school:
                professor_details = professor_details.filter(school_name=selected_school)
    
    context = {
        'schools': schools,
        'professors': filtered_professors,
        'selected_school': selected_school,
        'selected_professor': selected_professor,
        'professor_details': professor_details,
    }
    
    return render(request, 'show.html', context)

def professor_dropdown(request):
    # Get unique professor names for dropdown
    professors = ITEM.objects.values_list('professor_name', flat=True).distinct().order_by('professor_name')
    return render(request, 'professor_dropdown.html', {"professors": professors})

def professor_profile(request, professor_name):
    # Get all reviews for the specific professor
    reviews = ITEM.objects.filter(professor_name=professor_name)
    
    if not reviews.exists():
        return render(request, 'professor_profile.html', {
            'professor_name': professor_name,
            'error': 'Professor not found'
        })
    
    # Get professor statistics
    total_reviews = reviews.count()
    #average_rating = round(reviews.aggregate(avg_rating=models.Avg('star_rating'))['avg_rating'] or 0, 1)
    
    # Calculate differentially private average rating
    ratings = list(reviews.values_list('star_rating', flat=True))
    min_rating = 0.0
    max_rating = 5.0
    epsilon = 1.0  # Privacy loss
    noisy_avg, true_avg = dp_average(ratings, min_rating, max_rating, epsilon)
    average_rating = round(noisy_avg, 1)
    
    #average_difficulty = round(reviews.aggregate(avg_diff=models.Avg('difficulty'))['avg_diff'] or 0, 1)
    # Calculate differentially private average difficulty
    difficulty = list(reviews.values_list('difficulty', flat=True))
    min_difficulty = 1.0
    max_difficulty = 5.0
    epsilon = 1.0  # Privacy loss
    noisy_avg, true_avg = dp_difficulty_average(difficulty, min_difficulty, max_difficulty, epsilon)
    average_difficulty = round(noisy_avg, 1)

    # Calculate average help_useful
    # Using proper Django aggregate syntax
    # avg_result = reviews.aggregate(avg_help=models.Avg('help_useful'))
    # average_help_useful = round(avg_result.get('avg_help') or 0, 1)
    helpful = list(reviews.values_list('help_useful', flat=True))
    min_helpful = 1.0
    max_helpful = 10.0
    epsilon = 1.0  # Privacy loss
    noisy_avg, true_avg = dp_helpful_average(helpful, min_helpful, max_helpful, epsilon)
    average_help_useful = round(noisy_avg, 1)
    
    # Calculate percentage who would take again
    # would_take_again_count = reviews.filter(would_take_agains=True).count()
    # would_take_again_percent = round((would_take_again_count / total_reviews) * 100) if total_reviews > 0 else 0
    
    # Calculate differentially private percentage who would take again
    true_count = reviews.filter(would_take_agains=True).count()
    epsilon = 1.0  # Privacy loss
    noisy_count = dp_count(true_count, epsilon)
    # Ensure noisy_count is non-negative
    noisy_count = max(0, noisy_count)
    # Calculate noisy percentage
    would_take_again_percent = round((noisy_count / total_reviews) * 100) if total_reviews > 0 else 0
    # Ensure percentage is between 0 and 100
    would_take_again_percent = max(0, min(100, would_take_again_percent))
    
    # Get school name (assuming all reviews are from the same school)
    school_name = reviews.first().school_name
    department_name = reviews.first().department_name
    
    context = {
        'professor_name': professor_name,
        'school_name': school_name,
        'department_name':department_name,
        'reviews': reviews,
        'total_reviews': total_reviews,
        'average_rating': average_rating,
        'average_difficulty': average_difficulty,
        'average_help_useful': average_help_useful,
        'would_take_again_percent': would_take_again_percent,
    }
    
    return render(request, 'professor_profile.html', context)

def search_prof(request):
    search_query = request.GET.get('q', '').strip()
    professor_results = []
    debug_info = []
    
    if search_query:
        # Debug: Show what we're searching for
        debug_info.append(f"Searching for: '{search_query}'")
        
        # Debug: Show sample professor names from database
        sample_professors = ITEM.objects.values_list('professor_name', flat=True).distinct()[:3]
        debug_info.append(f"Sample names in DB: {list(sample_professors)}")
        
        # Check if it's a full name (contains space) or partial name
        if ' ' in search_query:
            debug_info.append("Full name search detected")
            
            # Try multiple search approaches for full names
            # 1. Exact match
            professors = ITEM.objects.filter(
                professor_name__iexact=search_query
            ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
            debug_info.append(f"Exact match results: {list(professors)}")
            
            # 2. If no exact match, try with normalized spacing
            if not professors.exists():
                # Normalize the search query (remove extra spaces)
                normalized_query = ' '.join(search_query.split())
                debug_info.append(f"Trying normalized query: '{normalized_query}'")
                professors = ITEM.objects.filter(
                    professor_name__iexact=normalized_query
                ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
                debug_info.append(f"Normalized exact match results: {list(professors)}")
            
            # 3. If still no match, try searching with double space (common issue)
            if not professors.exists():
                debug_info.append("Trying with double space")
                double_space_query = search_query.replace(' ', '  ')
                professors = ITEM.objects.filter(
                    professor_name__iexact=double_space_query
                ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
                debug_info.append(f"Double space search results: {list(professors)}")
            
            # 4. If still no match, try partial match
            if not professors.exists():
                debug_info.append("No exact match, trying partial match")
                professors = ITEM.objects.filter(
                    professor_name__icontains=search_query
                ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
                debug_info.append(f"Partial match results: {list(professors)}")
            
            # 5. If still no match, try searching for each part separately
            if not professors.exists():
                debug_info.append("Trying individual name parts")
                first_name, last_name = search_query.split(' ', 1)
                professors = ITEM.objects.filter(
                    professor_name__icontains=first_name
                ).filter(
                    professor_name__icontains=last_name
                ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
                debug_info.append(f"Individual parts search results: {list(professors)}")
        else:
            debug_info.append("Partial name search detected")
            # Partial name search - find all professors containing this name
            professors = ITEM.objects.filter(
                professor_name__icontains=search_query
            ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
            debug_info.append(f"Partial search results: {list(professors)}")
        
        # If we find exactly one professor, redirect directly to their profile
        if professors.count() == 1:
            return redirect('professor_profile', professor_name=professors.first())
        
        # Get detailed information for multiple professors
        for professor_name in professors:
            reviews = ITEM.objects.filter(professor_name=professor_name)
            if reviews.exists():
                # Get professor statistics
                total_reviews = reviews.count()
                #average_rating = round(reviews.aggregate(avg_rating=models.Avg('star_rating'))['avg_rating'] or 0, 1)
                
                # Calculate differentially private average rating
                ratings = list(reviews.values_list('star_rating', flat=True))
                min_rating = 0.0
                max_rating = 5.0
                epsilon = 1.0  # Privacy budget
                noisy_avg, true_avg = dp_average(ratings, min_rating, max_rating, epsilon)
                average_rating = round(noisy_avg, 1)
                
                #average_difficulty = round(reviews.aggregate(avg_diff=models.Avg('difficulty'))['avg_diff'] or 0, 1)
                # Calculate differentially private average difficulty
                difficulty = list(reviews.values_list('difficulty', flat=True))
                min_difficulty = 1.0
                max_difficulty = 5.0
                epsilon = 1.0  # Privacy budget
                noisy_avg, true_avg = dp_difficulty_average(difficulty, min_difficulty, max_difficulty, epsilon)
                average_difficulty = round(noisy_avg, 1)

                # Calculate average help_useful
                # Using proper Django aggregate syntax
                # avg_result = reviews.aggregate(avg_help=models.Avg('help_useful'))
                # average_help_useful = round(avg_result.get('avg_help') or 0, 1)
                helpful = list(reviews.values_list('help_useful', flat=True))
                min_helpful = 1.0
                max_helpful = 10.0
                epsilon = 1.0  # Privacy loss
                noisy_avg, true_avg = dp_helpful_average(helpful, min_helpful, max_helpful, epsilon)
                average_help_useful = round(noisy_avg, 1)
                
                # Calculate percentage who would take again
                # would_take_again_count = reviews.filter(would_take_agains=True).count()
                # would_take_again_percent = round((would_take_again_count / total_reviews) * 100) if total_reviews > 0 else 0
                # Calculate differentially private percentage who would take again
                true_count = reviews.filter(would_take_agains=True).count()
                epsilon = 1.0  # Privacy budget
                noisy_count = dp_count(true_count, epsilon)
                # Ensure noisy_count is non-negative
                noisy_count = max(0, noisy_count)
                # Calculate noisy percentage
                would_take_again_percent = round((noisy_count / total_reviews) * 100) if total_reviews > 0 else 0
                # Ensure percentage is between 0 and 100
                would_take_again_percent = max(0, min(100, would_take_again_percent))
                
                # Get school name
                school_name = reviews.first().school_name
                
                professor_results.append({
                    'name': professor_name,
                    'school_name': school_name,
                    'total_reviews': total_reviews,
                    'average_rating': average_rating,
                    'average_difficulty': average_difficulty,
                    'average_help_useful': average_help_useful,
                    'would_take_again_percent': would_take_again_percent,
                    'reviews': reviews[:3]  # Show first 3 reviews as preview
                })
    
    context = {
        'search_query': search_query,
        'professor_results': professor_results,
        'has_results': len(professor_results) > 0 if search_query else False,
        'debug_info': debug_info,
    }
    
    return render(request, 'search_prof.html', context)


def WriteReview(request,professor_name):
    # Render the write-review page for a specific professor and handle submission
    professor = ITEM.objects.filter(professor_name=professor_name).first()
    school_name = professor.school_name if professor else ''
    department_name = professor.department_name if professor else ''

    if request.method == 'POST':
        # Extract form values
        course = request.POST.get('course', '').strip()
        difficulty_raw = request.POST.get('difficulty', '').strip()
        help_useful_raw = request.POST.get('help_useful', '').strip()
        rating_raw = request.POST.get('rating', '').strip()
        would_take_raw = request.POST.get('would_take_agains', '').strip()
        comments = request.POST.get('message', '').strip()

        # Basic validation and type coercion
        try:
            difficulty = int(difficulty_raw) if difficulty_raw else None
        except ValueError:
            difficulty = None
        try:
            help_useful = int(help_useful_raw) if help_useful_raw else None
        except ValueError:
            help_useful = None
        if help_useful is not None:
            # Clamp to keep within allowed positive range (1-10)
            help_useful = max(1, min(10, help_useful))
        try:
            star_rating = float(rating_raw) if rating_raw else None
        except ValueError:
            star_rating = None
        would_take_agains = True if would_take_raw == 'true' else False if would_take_raw == 'false' else None

        # Minimal required fields check
        missing_fields = []
        if not course:
            missing_fields.append('course')
        if difficulty is None:
            missing_fields.append('difficulty')
        if help_useful is None:
            missing_fields.append('help_useful')
        if star_rating is None:
            missing_fields.append('rating')
        if would_take_agains is None:
            missing_fields.append('would_take_agains')
        if not comments:
            missing_fields.append('message')

        if missing_fields:
            messages.error(request, f"Please fill all required fields: {', '.join(missing_fields)}")
        else:
            # Check if user already used the rephrased version from frontend
            is_rephrased = request.POST.get('is_rephrased', '0') == '1'
            
            # Only apply make_review_private if user hasn't already used the rephrased version
            if is_rephrased:
                # User explicitly chose the rephrased version, use it as-is
                cleaned_comments = comments
            else:
                # Clean the comment for privacy before saving
                cleaned_comments = make_review_private(comments)

            # Ensure school_name and department_name are not empty
            # If professor doesn't exist, we need at least some default values
            if not school_name:
                school_name = 'Unknown'
            if not department_name:
                department_name = 'Unknown'

            # Create a new ITEM review entry
            try:
                ITEM.objects.create(
                    professor_name=professor_name,
                    school_name=school_name,
                    department_name=department_name,
                    star_rating=star_rating,
                    course=course,
                    difficulty=difficulty,
                    would_take_agains=would_take_agains if would_take_agains is not None else False,
                    help_useful=help_useful if help_useful is not None else 0,
                    comments=cleaned_comments,
                )
                messages.success(request, 'Your review has been submitted.')
                return redirect('professor_profile', professor_name=professor_name)
            except Exception as e:
                messages.error(request, f'Error saving review: {str(e)}')
                # Don't redirect, stay on the form so user can see the error

    # Build unique course list for this professor
    courses = list(
        ITEM.objects.filter(professor_name=professor_name)
        .values_list('course', flat=True)
        .distinct()
        .order_by('course')
    )
    context = {
        'professor_name': professor_name,
        'school_name': school_name,
        'department_name':department_name,
        'courses': courses,
    }
    return render(request,'write_review.html',context)

def WriteReviewBlank(request):
    # If a query is provided and looks like a full name, try to redirect directly
    search_query = request.GET.get('q', '').strip()
    if search_query and ' ' in search_query:
        professors = ITEM.objects.filter(
            professor_name__iexact=search_query
        ).values_list('professor_name', flat=True).distinct().order_by('professor_name')
        if professors.count() == 1:
            return redirect('WriteReview', professor_name=professors.first())
    # Fallback to home if no direct match
    return render(request,'home.html')

def Databaseshow(request):
    items = ITEM.objects.all().order_by('-id')
    return render(request,'databaseshow.html', { 'items': items })

def delete_review(request, review_id):
    if request.method == 'POST':
        ITEM.objects.filter(id=review_id).delete()
        messages.success(request, 'Review deleted.')
    return redirect('Databaseshow')
    