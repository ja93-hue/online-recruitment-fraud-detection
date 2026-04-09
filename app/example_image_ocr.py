"""
Example: Image-based Job Fraud Detection using OCR

This script demonstrates how to use the ImageOCR module to analyze
job posting screenshots or images for potential fraud with detailed analysis.

Usage:
    python example_image_ocr.py path/to/image.png           # Basic analysis
    python example_image_ocr.py path/to/image.png --detailed # Detailed analysis
    python example_image_ocr.py                              # Interactive mode
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import JobFraudDetector
from src.image_ocr import ImageOCR, predict_from_image


def analyze_image(image_path: str, detailed: bool = False):
    """
    Analyze a job posting image for fraud.
    
    Args:
        image_path: Path to the image file
        detailed: If True, include LIME explanations and full advisories
    """
    print("=" * 70)
    print("Job Fraud Detection - Image Analysis")
    print("=" * 70)
    print()
    
    # Initialize detector
    print("1. Initializing JobFraudDetector...")
    detector = JobFraudDetector()
    
    # Load model
    print("2. Loading trained model...")
    try:
        detector.load_model()
        print("   Model loaded successfully!")
    except FileNotFoundError:
        print("   ERROR: Model not found. Please train the model first:")
        print("   python train_bert.py")
        return
    
    # Initialize OCR
    print("3. Initializing OCR engine...")
    try:
        ocr = ImageOCR()
        print("   OCR engine ready!")
    except EnvironmentError as e:
        print(f"   ERROR: {e}")
        return
    
    # Analyze image
    print(f"\n4. Analyzing image: {image_path}")
    if detailed:
        print("   Mode: DETAILED ANALYSIS (with LIME explanations)")
    print("-" * 70)
    
    try:
        # Choose analysis method based on mode
        if detailed:
            result = ocr.predict_from_image_detailed(
                image_path, 
                detector,
                include_lime=True,
                lime_num_features=10
            )
        else:
            result = ocr.predict_from_image(image_path, detector)
        
        # Display results
        print(f"\n{'='*70}")
        print("ANALYSIS RESULTS")
        print("=" * 70)
        
        if result.get('error'):
            print(f"Error: {result['error']}")
            return
        
        # Check if detailed analysis summary is available
        if 'analysis_summary' in result:
            print(result['analysis_summary'])
        else:
            # Basic result display
            label = result['label']
            confidence = result['confidence']
            
            if label == 'Fraudulent':
                print(f"\n⚠️  PREDICTION: {label}")
            else:
                print(f"\n✅ PREDICTION: {label}")
            
            print(f"   Confidence: {confidence}%")
            
            # Probabilities
            print(f"\n📊 Probabilities:")
            print(f"   Legitimate: {result['probabilities']['legitimate']}%")
            print(f"   Fraudulent: {result['probabilities']['fraudulent']}%")
            
            # Fraud signals
            if 'fraud_signals' in result and result['fraud_signals']:
                print(f"\n🚨 Detected Red Flags:")
                for signal in result['fraud_signals']:
                    print(f"   - {signal}")
        
        # Display detailed advisories if available
        if detailed and 'posting_details' in result:
            details = result['posting_details']
            
            # Detailed advisories
            if details.get('detailed_advisories'):
                print(f"\n{'='*70}")
                print("DETAILED ADVISORIES")
                print("=" * 70)
                
                for i, adv in enumerate(details['detailed_advisories'], 1):
                    icon = adv.get('icon', '•')
                    print(f"\n{icon} Advisory {i}: {adv['finding']}")
                    print(f"   Category: {adv['category']}")
                    print(f"   Risk Level: {adv['risk_level']}")
                    print(f"\n   {'-'*60}")
                    # Wrap advisory text nicely
                    advisory_text = adv.get('advisory', '')
                    for line in advisory_text.split('\n'):
                        print(f"   {line}")
            
            # Extracted info
            if details.get('extracted_info'):
                print(f"\n{'='*70}")
                print("EXTRACTED INFORMATION")
                print("=" * 70)
                for key, value in details['extracted_info'].items():
                    print(f"   {key}: {value}")
        
        # LIME explanation details
        if detailed and 'lime_explanation' in result:
            lime = result['lime_explanation']
            
            if lime.get('interpretation'):
                print(f"\n{'='*70}")
                print("LIME EXPLANATION")
                print("=" * 70)
                print(f"\n{lime['interpretation']}")
            
            if lime.get('key_indicators'):
                print("\n📊 Words Supporting This Prediction:")
                for ind in lime['key_indicators']:
                    bar = "█" * int(abs(ind['weight']) * 50)
                    print(f"   \"{ind['word']}\" {bar}")
            
            if lime.get('opposing_indicators'):
                print("\n📉 Words Opposing This Prediction:")
                for ind in lime['opposing_indicators']:
                    bar = "░" * int(abs(ind['weight']) * 50)
                    print(f"   \"{ind['word']}\" {bar}")
        
        # OCR quality
        if 'ocr_confidence' in result:
            ocr_conf = result['ocr_confidence']
            print(f"\n📷 OCR Quality: {ocr_conf.get('quality', 'unknown')}")
            if ocr_conf.get('average_score'):
                print(f"   Average confidence: {ocr_conf['average_score']:.1f}%")
        
        # BERT raw scores
        if 'bert_raw' in result:
            print(f"\n🤖 BERT Model Raw Scores:")
            print(f"   Legitimate: {result['bert_raw']['legitimate']}%")
            print(f"   Fraudulent: {result['bert_raw']['fraudulent']}%")
        
        # Extracted text preview
        text = result.get('extracted_text', '')
        if text:
            print(f"\n{'='*70}")
            print(f"EXTRACTED TEXT ({result['text_length']} characters)")
            print("=" * 70)
            # Show first 500 characters
            preview = text[:500]
            if len(text) > 500:
                preview += "\n... [truncated]"
            print(preview)
        
        print("\n" + "=" * 70)
        
    except FileNotFoundError:
        print(f"ERROR: Image file not found: {image_path}")
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Run in interactive mode, prompting user for image path."""
    print("\n" + "=" * 70)
    print("Image-based Job Fraud Detection - Interactive Mode")
    print("=" * 70)
    print("\nSupported formats: PNG, JPG, JPEG, BMP, TIFF, WebP, GIF")
    print("Commands:")
    print("  <path>           - Analyze image (basic)")
    print("  <path> --detailed - Analyze image with full LIME explanation")
    print("  q/quit           - Exit")
    print()
    
    # Initialize once for multiple images
    print("Initializing...")
    detector = JobFraudDetector()
    
    try:
        detector.load_model()
    except FileNotFoundError:
        print("ERROR: Model not found. Please train the model first:")
        print("python train_bert.py")
        return
    
    try:
        ocr = ImageOCR()
    except EnvironmentError as e:
        print(f"ERROR: {e}")
        return
    
    print("Ready!\n")
    
    while True:
        user_input = input("Enter image path (or 'q' to quit): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Check for --detailed flag
        detailed = '--detailed' in user_input or '-d' in user_input
        image_path = user_input.replace('--detailed', '').replace('-d', '').strip()
        
        # Remove quotes if present
        image_path = image_path.strip('"\'')
        
        if not Path(image_path).exists():
            print(f"File not found: {image_path}\n")
            continue
        
        try:
            if detailed:
                print("\n🔍 Running detailed analysis (this may take a moment)...")
                result = ocr.predict_from_image_detailed(image_path, detector)
                
                print("\n" + "=" * 70)
                if result.get('error'):
                    print(f"Error: {result['error']}")
                elif 'analysis_summary' in result:
                    print(result['analysis_summary'])
                    
                    # Show detailed advisories
                    if 'posting_details' in result:
                        details = result['posting_details']
                        advisories = details.get('detailed_advisories', [])
                        if advisories:
                            print(f"\n📋 Detailed Advisories:")
                            for adv in advisories:
                                icon = adv.get('icon', '•')
                                print(f"   {icon} [{adv['risk_level']}] {adv['finding']}")
                    
                    # Show LIME keywords
                    if 'lime_explanation' in result:
                        lime = result['lime_explanation']
                        if lime.get('key_indicators'):
                            print(f"\n🔑 Key Words:")
                            for ind in lime['key_indicators'][:5]:
                                print(f"   • \"{ind['word']}\"")
                print("=" * 70 + "\n")
            else:
                result = ocr.predict_from_image(image_path, detector)
                
                print("\n" + "-" * 50)
                if result.get('error'):
                    print(f"Error: {result['error']}")
                else:
                    label = result['label']
                    confidence = result['confidence']
                    
                    if label == 'Fraudulent':
                        print(f"⚠️  {label} ({confidence}% confidence)")
                    else:
                        print(f"✅ {label} ({confidence}% confidence)")
                    
                    if 'fraud_signals' in result:
                        print("Red flags:", ", ".join(result['fraud_signals']))
                    
                    print(f"Text extracted: {result['text_length']} characters")
                    print("\nTip: Use '--detailed' for full analysis with explanations")
                print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Check for --detailed flag
        detailed = '--detailed' in sys.argv or '-d' in sys.argv
        
        # Get image path (first non-flag argument)
        image_path = None
        for arg in sys.argv[1:]:
            if not arg.startswith('-'):
                image_path = arg
                break
        
        if image_path:
            analyze_image(image_path, detailed=detailed)
        else:
            print("Usage: python example_image_ocr.py <image_path> [--detailed]")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
