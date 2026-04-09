"""
=============================================================================
Comprehensive Test Suite for Fake Job Detection System
=============================================================================

Tests the hybrid detection system (Rule-Based + BERT) with various examples:
1. Legitimate job postings (should pass)
2. Obvious scams (should flag)
3. Subtle/sophisticated scams (key test for BERT)

Run with: python -m pytest tests/test_detection.py -v
Or directly: python tests/test_detection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import JobFraudDetector

# =============================================================================
# TEST CASES
# =============================================================================

LEGITIMATE_JOBS = [
    {
        "name": "Software Engineer at Google",
        "text": """
        Software Engineer - Full Stack
        
        Company: Google LLC
        Location: Mountain View, CA (Hybrid)
        
        About the Role:
        We're looking for a skilled Full Stack Engineer to join our Cloud team. You'll work on
        large-scale distributed systems serving millions of users worldwide.
        
        Requirements:
        - BS/MS in Computer Science or equivalent experience
        - 3+ years of experience with Python, Java, or Go
        - Experience with cloud platforms (GCP, AWS, Azure)
        - Strong problem-solving skills
        
        Benefits:
        - Competitive salary ($150,000 - $250,000)
        - Stock options and bonuses
        - Health, dental, vision insurance
        - 401(k) matching
        - Unlimited PTO
        
        To apply, submit your resume through our careers portal at careers.google.com
        """,
        "expected": "legitimate"
    },
    {
        "name": "Marketing Manager at Nike",
        "text": """
        Marketing Manager - Brand Strategy
        
        Nike, Inc. | Portland, OR
        
        Join our world-class marketing team to drive brand engagement for one of the most
        recognizable brands in the world.
        
        What You'll Do:
        - Develop and execute marketing campaigns
        - Analyze market trends and consumer behavior
        - Collaborate with creative teams on brand messaging
        - Manage marketing budget of $2M+
        
        What You Bring:
        - 5+ years in brand marketing
        - MBA preferred
        - Experience with digital marketing platforms
        - Strong analytical and communication skills
        
        We offer competitive compensation, employee discounts, and comprehensive benefits.
        
        Apply at nike.com/careers
        Reference: MKT-2024-0892
        """,
        "expected": "legitimate"
    },
    {
        "name": "Data Analyst Entry Level",
        "text": """
        Junior Data Analyst
        
        TechCorp Solutions - Austin, TX
        
        We're hiring a Junior Data Analyst to join our growing analytics team. This is a great
        entry-level opportunity for recent graduates.
        
        Responsibilities:
        - Assist senior analysts with data cleaning and preparation
        - Create reports and visualizations using Tableau
        - Support business teams with ad-hoc analysis requests
        
        Requirements:
        - Bachelor's degree in Statistics, Math, or related field
        - Proficiency in Excel and SQL
        - Basic Python or R knowledge a plus
        - Strong attention to detail
        
        Salary: $55,000 - $65,000 + benefits
        
        Interview process includes phone screen, technical assessment, and onsite interview.
        
        Contact: hr@techcorpsolutions.com
        """,
        "expected": "legitimate"
    }
]

OBVIOUS_SCAMS = [
    {
        "name": "Gift Card Reshipping Scam",
        "text": """
        URGENT!!! Work From Home - Earn $5000/Week!!!
        
        NO EXPERIENCE NEEDED! NO INTERVIEW REQUIRED!
        
        We need people to help process orders from home. Simply receive packages,
        purchase gift cards with the provided funds, and send the gift card codes to
        our processing center.
        
        BENEFITS:
        - $500 PER DAY!!!
        - Work your own hours
        - Start IMMEDIATELY
        
        To get started, email your SSN, bank account details, and a copy of your
        driver's license to jobs@quickcash-work.xyz
        
        ACT NOW - LIMITED POSITIONS AVAILABLE!!!
        """,
        "expected": "fraud"
    },
    {
        "name": "Upfront Fee Scam",
        "text": """
        Data Entry Specialist - Remote
        
        GlobalData Inc.
        
        Looking for motivated individuals to join our data entry team!
        
        Requirements:
        - Computer with internet
        - Attention to detail
        
        Pay: $25/hour
        
        IMPORTANT: Before starting, all new hires must purchase our proprietary
        software license ($299) and complete the mandatory training course ($149).
        These fees will be reimbursed after your first month!
        
        Contact: hiring.manager2847@gmail.com
        WhatsApp: +1-555-0123
        """,
        "expected": "fraud"
    },
    {
        "name": "Check Cashing Scam",
        "text": """
        Financial Processing Agent - Work From Home
        
        Earn money processing financial transactions!
        
        Job duties:
        - Receive checks in your name
        - Deposit checks to your personal bank account
        - Wire 90% of funds to our overseas office
        - Keep 10% as your commission!
        
        No experience needed. Start earning within 24 hours!
        
        Reply to this posting with your full name, address, and bank routing number.
        
        Telegram: @quickmoney_jobs
        """,
        "expected": "fraud"
    }
]

SUBTLE_SCAMS = [
    {
        "name": "NexaCore Onboarding Kit Scam",
        "text": """
        Remote Operations Assistant
        
        Company: NexaCore Technologies
        Location: Remote (India)
        Employment Type: Full-time
        Salary: ₹35,000 - ₹50,000 per month
        
        Company Description:
        NexaCore Technologies is a rapidly growing tech startup specializing in cloud-based
        business solutions. We are expanding our remote operations team and seeking dedicated
        professionals to join our innovative workplace.
        
        Role Description:
        As a Remote Operations Assistant, you will support our project management team with
        administrative tasks, data entry, client communication, and workflow coordination.
        This is an excellent opportunity for individuals looking to build a career in tech operations.
        
        Requirements:
        - Bachelor's degree in any field
        - Strong communication skills in English
        - Basic computer proficiency (MS Office, Google Workspace)
        - Reliable internet connection
        - Self-motivated and detail-oriented
        
        Benefits:
        - Competitive salary with performance bonuses
        - Flexible work hours
        - Career growth opportunities
        - Health insurance after probation period
        
        Application Process:
        Interested candidates should submit their resume and a brief cover letter.
        Shortlisted applicants will receive a detailed onboarding guide.
        
        Note: Selected candidates will be required to purchase a proprietary onboarding kit
        (₹4,500), which includes software access and training materials. This cost will be
        reimbursed after 3 months of successful employment.
        """,
        "expected": "fraud"
    },
    {
        "name": "Fake Consulting Firm",
        "text": """
        Business Development Associate
        
        Apex Global Consulting
        Location: Remote
        
        Join our dynamic consulting firm! We work with Fortune 500 clients across industries.
        
        About Us:
        Apex Global Consulting provides strategic advisory services to businesses worldwide.
        Our team of experts helps companies optimize operations and drive growth.
        
        The Role:
        - Research potential clients and market opportunities
        - Prepare presentation materials
        - Coordinate with senior consultants
        - Manage client communications
        
        Requirements:
        - Degree in Business or related field
        - Excellent English communication
        - Proficient in PowerPoint and Excel
        
        Compensation: $4,000 - $6,000/month
        
        Onboarding Process:
        To ensure quality hires, candidates must complete our certified assessment program
        ($350). Top performers receive full reimbursement within 60 days. Contact our HR
        team via WhatsApp at +44-7XXX-XXXXXX to begin the process.
        """,
        "expected": "fraud"
    },
    {
        "name": "Crypto Investment Scam Disguised as Job",
        "text": """
        Investment Research Analyst
        
        FinTech Innovations Ltd
        Remote | Part-time | $30-50/hr
        
        We're seeking detail-oriented individuals to join our investment research team.
        
        Responsibilities:
        - Monitor cryptocurrency market trends
        - Analyze blockchain transaction patterns
        - Prepare daily market reports
        - Test our proprietary trading algorithms
        
        Requirements:
        - Interest in financial markets
        - Basic understanding of cryptocurrency
        - Ability to work independently
        
        Getting Started:
        As part of training, analysts will practice live trading using company funds.
        New hires receive a starter portfolio of $500 in crypto assets. To unlock your
        trading account, a refundable security deposit of $200 is required.
        
        Email: hr@fintechinnovations-careers.net
        """,
        "expected": "fraud"
    },
    {
        "name": "MLM Disguised as Marketing Job",
        "text": """
        Independent Marketing Representative
        
        LiveWell Products International
        
        Are you entrepreneurial and self-motivated? Join our team of successful
        marketing representatives!
        
        What We Offer:
        - Unlimited earning potential ($2,000 - $10,000/month)
        - Be your own boss
        - Flexible schedule
        - Comprehensive training and support
        
        Your Role:
        - Promote our premium health and wellness products
        - Build and manage your own team
        - Attend weekly virtual training sessions
        - Achieve monthly sales targets
        
        Investment:
        To get started, representatives purchase a starter kit ($499) containing
        product samples and marketing materials. Most successful reps earn back
        their investment within the first two weeks!
        
        Contact: success@livewellproducts.biz
        """,
        "expected": "fraud"
    },
    {
        "name": "Shipping Repackaging Scam",
        "text": """
        Logistics Coordinator - Work From Home
        
        Premier Logistics Solutions
        
        We're expanding our home-based logistics network and need reliable coordinators.
        
        Position Overview:
        - Receive shipments at your home address
        - Inspect and repackage items
        - Arrange forwarding to international destinations
        - Maintain shipping records
        
        Requirements:
        - Dedicated home office space
        - Available during business hours
        - Attention to detail
        - Reliable transportation (for occasional pickups)
        
        Compensation:
        - $400/week base salary
        - $15 per package processed
        - Performance bonuses
        
        Note: Initial training period requires use of personal funds for shipping
        supplies (approximately $150). All expenses are reimbursed within 2 weeks.
        
        Apply via Telegram: @PremierLogisticsHR
        """,
        "expected": "fraud"
    }
]

# Additional edge cases
EDGE_CASES = [
    {
        "name": "Legitimate Remote Job with Relaxed Policies",
        "text": """
        Content Writer - Remote
        
        BlogMedia Inc. | Fully Remote
        
        We're a content marketing agency looking for talented writers to join our team.
        
        What You'll Do:
        - Write blog posts, articles, and social media content
        - Research topics across various industries
        - Edit and proofread content
        - Meet weekly deadlines
        
        Requirements:
        - Portfolio of writing samples
        - Excellent grammar and research skills
        - Ability to write 2000+ words daily
        
        Compensation: $0.10/word (most writers earn $3,000-5,000/month)
        
        Our hiring process is simple - submit samples, complete a paid trial article,
        and if it's a fit, you're in! No lengthy interviews.
        
        Email: writers@blogmediainc.com
        Website: www.blogmediainc.com
        """,
        "expected": "legitimate"
    },
    {
        "name": "Startup with Equity Compensation",
        "text": """
        Early Stage Startup - Full Stack Developer
        
        Stealth Mode AI Startup | San Francisco / Remote
        
        We're building the next generation of AI-powered productivity tools and looking
        for a talented full stack developer to join as employee #3.
        
        The Opportunity:
        - Ground floor opportunity at a well-funded startup
        - Work directly with founders (ex-Google, ex-Meta)
        - Shape product direction and company culture
        
        Tech Stack:
        - Python, FastAPI, PostgreSQL
        - React, TypeScript
        - AWS/GCP
        
        Compensation:
        - Base: $120k-150k (flexible based on experience)
        - Equity: 0.5-1.5% (4-year vest, 1-year cliff)
        - Full benefits, unlimited PTO
        
        Process: Intro call → Technical interview → Founder chat → Offer
        
        Interested? Email your resume and GitHub to founders@stealthai.co
        """,
        "expected": "legitimate"
    }
]


def run_tests():
    """Run all test cases and report results."""
    print("=" * 70)
    print("FAKE JOB DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Initialize the detector
    print("\n[1/5] Loading BERT model...")
    try:
        detector = JobFraudDetector()
        detector.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nFalling back to rule-based detection only...")
        detector = None
    
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    all_tests = [
        ("LEGITIMATE JOBS", LEGITIMATE_JOBS),
        ("OBVIOUS SCAMS", OBVIOUS_SCAMS),
        ("SUBTLE/SOPHISTICATED SCAMS", SUBTLE_SCAMS),
        ("EDGE CASES", EDGE_CASES)
    ]
    
    for category_idx, (category_name, test_cases) in enumerate(all_tests, 2):
        print(f"\n[{category_idx}/5] Testing {category_name}...")
        print("-" * 50)
        
        for test in test_cases:
            name = test["name"]
            text = test["text"]
            expected = test["expected"]
            
            try:
                if detector:
                    result = detector.predict(text)
                    is_fraud = result['prediction'] == 1
                    confidence = result['confidence']
                    fraud_signals = result.get('fraud_signals', [])
                    bert_raw = result.get('bert_raw', {})
                else:
                    # Fallback: basic keyword detection
                    is_fraud = any(kw in text.lower() for kw in 
                                  ['gift card', 'wire', 'western union', 'act now', 
                                   'no experience needed', 'ssn', 'bank account'])
                    confidence = 80 if is_fraud else 20
                    fraud_signals = []
                    bert_raw = {}
                
                actual = "fraud" if is_fraud else "legitimate"
                passed = actual == expected
                
                if passed:
                    results["passed"] += 1
                    status = "✓ PASS"
                else:
                    results["failed"] += 1
                    status = "✗ FAIL"
                
                print(f"\n  {status}: {name}")
                print(f"    Expected: {expected.upper()} | Got: {actual.upper()} ({confidence:.1f}%)")
                
                if bert_raw:
                    print(f"    BERT Raw: {bert_raw.get('fraudulent', 'N/A')}% fraud")
                
                if fraud_signals:
                    print(f"    Rule Signals: {', '.join(fraud_signals[:3])}")
                
                results["details"].append({
                    "name": name,
                    "category": category_name,
                    "expected": expected,
                    "actual": actual,
                    "confidence": confidence,
                    "passed": passed,
                    "signals": fraud_signals
                })
                
            except Exception as e:
                print(f"\n  ✗ ERROR: {name}")
                print(f"    {str(e)}")
                results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total = results["passed"] + results["failed"]
    pass_rate = (results["passed"] / total * 100) if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {results['passed']} ({pass_rate:.1f}%)")
    print(f"Failed: {results['failed']}")
    
    if results["failed"] > 0:
        print("\n⚠️  Failed Tests:")
        for detail in results["details"]:
            if not detail["passed"]:
                print(f"  - {detail['name']} (expected {detail['expected']}, got {detail['actual']})")
    
    print("\n" + "=" * 70)
    
    # Return success/failure for CI
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
