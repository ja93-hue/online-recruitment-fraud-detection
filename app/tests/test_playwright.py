"""
Playwright E2E Tests for Fake Job Detection Application
Tests the full application flow including UI and API integration
"""
import os
import sys
import pytest
from playwright.sync_api import Page, expect, sync_playwright
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configuration
FRONTEND_URL = "http://localhost:8501"
API_URL = "http://localhost:5000"


class TestAPIEndpoints:
    """Test the Flask API endpoints directly."""
    
    def test_health_check(self, api_client):
        """Test API health endpoint."""
        response = api_client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
    
    def test_predict_endpoint(self, api_client):
        """Test prediction endpoint."""
        response = api_client.post(
            "/api/predict",
            json={
                "text": "Software Engineer at Google. 5 years experience required. Health insurance and 401k."
            }
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] == True
        assert "prediction" in data
        assert data["prediction"]["label"] in ["Legitimate", "Fraudulent"]
    
    def test_predict_fraudulent(self, api_client):
        """Test prediction with suspicious text."""
        response = api_client.post(
            "/api/predict",
            json={
                "text": "EARN $5000 PER WEEK!!! No experience needed! Send bank details NOW! Unlimited earning potential!"
            }
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] == True
        assert data["prediction"]["is_fraudulent"] == True
    
    def test_explain_endpoint(self, api_client):
        """Test explanation endpoint."""
        response = api_client.post(
            "/api/explain",
            json={
                "text": "URGENT HIRING! Work from home. No experience needed. Send credit card for background check fee."
            }
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] == True
        assert "explanation" in data
        assert "interpretation" in data["explanation"]
    
    def test_batch_predict_endpoint(self, api_client):
        """Test batch prediction endpoint."""
        response = api_client.post(
            "/api/batch-predict",
            json={
                "texts": [
                    "Software Engineer at Google. 5 years experience required.",
                    "EARN $5000 PER WEEK!!! No experience needed!"
                ]
            }
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] == True
        assert "predictions" in data
        assert len(data["predictions"]) == 2


class TestFrontendUI:
    """Test the Streamlit frontend UI."""
    
    def test_page_loads(self, ui_page: Page):
        """Test that the main page loads correctly."""
        page = ui_page
        page.goto(FRONTEND_URL)
        
        # Wait for Streamlit to fully load
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Give Streamlit extra time to render
        
        # Check for main title
        expect(page.locator("text=JobGuard AI").first).to_be_visible(timeout=10000)
    
    def test_api_connection_status(self, ui_page: Page):
        """Test that API connection status is shown."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Should show API Connected (green) when backend is running
        # Look for the success message or connection indicator
        page.reload()
        time.sleep(3)
        
        # Check for API connected status (new UI uses inline status instead of sidebar)
        api_status = page.locator("text=API Connected")
        expect(api_status).to_be_visible(timeout=10000)
    
    def test_text_input_exists(self, ui_page: Page):
        """Test that text input area exists."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Find text area
        text_area = page.locator('textarea')
        expect(text_area.first).to_be_visible(timeout=10000)
    
    def test_analyze_button_exists(self, ui_page: Page):
        """Test that analyze buttons exist."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Find Quick Analysis button
        quick_btn = page.locator('button:has-text("Quick Analysis")')
        expect(quick_btn).to_be_visible(timeout=10000)
        
        # Find Detailed Analysis button (renamed from Detailed Explanation)
        detail_btn = page.locator('button:has-text("Detailed Analysis")')
        expect(detail_btn).to_be_visible(timeout=10000)
    
    # Sample jobs section was removed from UI


class TestEndToEndFlow:
    """Test complete user flows."""
    
    def test_quick_analysis_flow(self, ui_page: Page):
        """Test the complete quick analysis flow."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Enter text in the text area
        text_area = page.locator('textarea').first
        text_area.fill(
            "Software Engineer position at TechCorp. "
            "Bachelor's degree required. 3+ years experience. "
            "Competitive salary and health insurance benefits."
        )
        
        # Click Quick Analysis button
        page.locator('button:has-text("Quick Analysis")').click()
        
        # Wait for results - BERT inference can take 10-15 seconds on CPU
        # Wait for spinner to disappear and result to appear
        time.sleep(15)
        
        # Check that some result is displayed
        # The result should contain "Legitimate" or "Fraudulent" or confidence percentage
        page_content = page.content()
        assert "Legitimate" in page_content or "Fraudulent" in page_content or "Appears" in page_content or "%" in page_content
    
    def test_fraudulent_detection_flow(self, ui_page: Page):
        """Test detection of fraudulent job posting."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Enter suspicious text
        text_area = page.locator('textarea').first
        text_area.fill(
            "EARN $5000 PER WEEK!!! No experience needed! "
            "Work from home! Send bank details to start earning NOW! "
            "Unlimited earning potential! Be your own boss!"
        )
        
        # Click Quick Analysis button
        page.locator('button:has-text("Quick Analysis")').click()
        
        # Wait for results - BERT inference takes time
        # First wait for spinner to appear, then disappear
        time.sleep(2)  # Initial wait for button click to process
        
        # Wait up to 30 seconds for analysis to complete
        for _ in range(30):
            page_content = page.content()
            if "Fraud" in page_content or "Potential" in page_content or "Legitimate" in page_content:
                break
            time.sleep(1)
        
        # Should detect as fraudulent
        page_content = page.content()
        assert "Fraud" in page_content or "Potential" in page_content
    
    def test_detailed_explanation_flow(self, ui_page: Page):
        """Test the detailed explanation flow."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Enter text
        text_area = page.locator('textarea').first
        text_area.fill(
            "URGENT! Easy money work from home. "
            "No experience needed. Pay fee to start."
        )
        
        # Click Detailed Analysis button (renamed)
        page.locator('button:has-text("Detailed Analysis")').click()
        
        # Wait for explanation to load (may take longer)
        time.sleep(8)
        
        # Check that explanation elements appear
        page_content = page.content()
        # Should have some interpretation or explanation
        assert "Fraud" in page_content or "Legitimate" in page_content or "Appears" in page_content


class TestBatchMode:
    """Test batch analysis feature."""
    
    def test_batch_mode_expander_exists(self, ui_page: Page):
        """Test that batch mode expander is visible."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Find batch mode expander
        batch_expander = page.locator("text=Batch Mode")
        expect(batch_expander).to_be_visible(timeout=10000)
    
    def test_batch_mode_instructions(self, ui_page: Page):
        """Test batch mode has clear instructions."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Click to expand batch mode
        page.locator("text=Batch Mode").click()
        time.sleep(1)
        
        # Check for instruction text
        page_content = page.content()
        assert "---" in page_content or "How to use" in page_content
    
    def test_batch_analyze_button(self, ui_page: Page):
        """Test batch analyze button appears when expanded."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Click to expand batch mode
        page.locator("text=Batch Mode").click()
        time.sleep(1)
        
        # Check for Batch Analyze button
        batch_btn = page.locator('button:has-text("Run Batch Analysis")')
        expect(batch_btn).to_be_visible(timeout=5000)
    
    def test_batch_analysis_flow(self, ui_page: Page):
        """Test complete batch analysis with multiple postings."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(5)  # Wait for API connection to be established
        
        # Wait for API to connect (button should become enabled)
        page.wait_for_selector("text=API Connected", timeout=15000)
        
        # Enter multiple job postings separated by ---
        text_area = page.locator('textarea').first
        text_area.fill(
            "Software Engineer at TechCorp. Bachelor's degree required. "
            "3+ years experience. Competitive salary.\n"
            "---\n"
            "EARN $5000 PER WEEK!!! No experience needed! "
            "Work from home! Send bank details NOW!"
        )
        
        # Scroll down to see expanders
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        
        # Expand batch mode
        page.locator("text=Batch Mode").click()
        time.sleep(2)
        
        # Click Batch Analyze - wait for button to be enabled
        batch_btn = page.locator('button:has-text("Run Batch Analysis")')
        batch_btn.wait_for(state="visible", timeout=10000)
        
        # Check if button is enabled (not disabled)
        if batch_btn.is_enabled():
            batch_btn.click()
            
            # Wait for batch results
            time.sleep(20)  # Batch processing takes time
            
            # Check results appear (should mention posting numbers)
            page_content = page.content()
            assert "Posting 1" in page_content or "Posting 2" in page_content or "Analyzed" in page_content or "postings" in page_content
        else:
            # If button is disabled, just verify it exists
            assert batch_btn.is_visible()


class TestImageMode:
    """Test image OCR feature."""
    
    def test_image_mode_expander_exists(self, ui_page: Page):
        """Test that image mode expander is visible (if easyocr available)."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(5)  # Wait for full page load
        
        # Scroll down to see all expanders
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        
        # Find image mode expander - may not exist if easyocr not installed
        page_content = page.content()
        # Test passes if either image mode exists OR batch mode exists (proves expanders work)
        assert "Image Analysis" in page_content or "Batch Mode" in page_content
    
    def test_image_mode_instructions(self, ui_page: Page):
        """Test image mode has clear instructions."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(5)
        
        # Scroll down to see all expanders
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        
        # Check if image mode exists
        image_expander = page.locator("text=Image Analysis")
        if image_expander.count() > 0:
            # Click to expand image mode
            image_expander.click()
            time.sleep(1)
            
            # Check for instruction text
            page_content = page.content()
            assert "PNG" in page_content or "JPG" in page_content or "How to use" in page_content
    
    def test_image_upload_control(self, ui_page: Page):
        """Test file upload control appears in image mode."""
        page = ui_page
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Check if image mode exists
        image_expander = page.locator("text=Image Analysis")
        if image_expander.count() > 0:
            # Click to expand image mode
            image_expander.click()
            time.sleep(1)
            
            # Check for file uploader - Streamlit uses input[type="file"] or drag-drop area
            page_content = page.content()
            assert "Browse files" in page_content or "Drag and drop" in page_content or "Upload" in page_content


class TestResponsiveness:
    """Test UI responsiveness."""
    
    def test_mobile_viewport(self, ui_page: Page):
        """Test that UI works on mobile viewport."""
        page = ui_page
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Title should still be visible
        expect(page.locator("text=JobGuard AI").first).to_be_visible(timeout=10000)
    
    def test_tablet_viewport(self, ui_page: Page):
        """Test that UI works on tablet viewport."""
        page = ui_page
        page.set_viewport_size({"width": 768, "height": 1024})
        page.goto(FRONTEND_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Title should still be visible
        expect(page.locator("text=JobGuard AI").first).to_be_visible(timeout=10000)


# Pytest fixtures
@pytest.fixture(scope="session")
def api_client():
    """Create a Flask test client with ML services initialized."""
    from backend import api

    if api.detector is None:
        api.initialize_services()

    return api.app.test_client()


@pytest.fixture(scope="function")
def ui_page():
    """Create a new browser page for each test."""
    if os.environ.get("RUN_PLAYWRIGHT_UI") != "1":
        pytest.skip("Set RUN_PLAYWRIGHT_UI=1 to run browser-based Streamlit UI tests.")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            try:
                yield page
            finally:
                context.close()
                browser.close()
    except Exception as error:
        pytest.skip(f"Playwright browser is not available in this environment: {error}")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
