#!/usr/bin/env python3
"""
UI Renderer and Executor

This module provides capabilities to:
- Execute and run applications
- Capture screenshots of UIs
- Analyze visual elements
- Test functionality
"""

import os
import time
import json
import base64
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import cv2
import numpy as np
from PIL import Image
import io

# Import core utilities for secure execution and configuration
from core import secure_executor, config, UIExecutionError, UIExecutionTimeoutError, UIExecutionFailedError, get_logger

logger = get_logger(__name__)

class UIRenderer:
    """Handles UI rendering, execution, and visual analysis"""
    
    def __init__(self, headless: Optional[bool] = None, timeout: Optional[int] = None):
        # Use config values if not provided
        self.headless = headless if headless is not None else config.ui.headless
        self.timeout = timeout if timeout is not None else config.ui.timeout
        self.driver: Optional[webdriver.Chrome] = None
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        # Ensure absolute path for screenshots
        self.screenshots_dir = self.screenshots_dir.resolve()
    
    def _find_available_port(self, start_port: int = 3000) -> int:
        """Find an available port starting from start_port"""
        import socket
        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        return start_port  # Fallback to original port
        
        logger.info(f"UIRenderer initialized: headless={self.headless}, timeout={self.timeout}")
    
    def setup_driver(self):
        """Setup Chrome WebDriver for UI testing"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            return True
        except Exception as e:
            print(f"Failed to setup WebDriver: {e}")
            return False
    
    def cleanup(self):
        """Clean up WebDriver resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories created during cloning"""
        try:
            import tempfile
            import shutil
            import glob
            
            # Find and remove temporary directories created by this instance
            temp_pattern = os.path.join(tempfile.gettempdir(), "ui_renderer_*")
            temp_dirs = glob.glob(temp_pattern)
            
            for temp_dir in temp_dirs:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up {temp_dir}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _clone_repository(self, repo_url: str) -> Optional[str]:
        """Clone a GitHub repository to a temporary directory"""
        try:
            import tempfile
            import shutil
            from pathlib import Path
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix="ui_renderer_")
            
            # Clone the repository (use relative path to avoid Windows issues)
            clone_result = secure_executor.execute_safe([
                "git", "clone", repo_url, os.path.basename(temp_dir)
            ], timeout=120, cwd=os.path.dirname(temp_dir))
            
            if not clone_result['success']:
                logger.error(f"Failed to clone repository: {clone_result['stderr']}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
            
            logger.info(f"Successfully cloned repository to {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return None
    
    def execute_web_app(self, project_path: str, port: Optional[int] = None) -> Dict[str, Any]:
        """Execute a web application and capture its UI"""
        result = {
            'success': False,
            'url': None,
            'screenshots': [],
            'errors': [],
            'analysis': {}
        }
        
        try:
            # Check if project_path is a GitHub URL
            if project_path.startswith('https://github.com/'):
                # Clone the repository first
                cloned_path = self._clone_repository(project_path)
                if not cloned_path:
                    result['errors'].append('Failed to clone repository')
                    return result
                project_path = cloned_path
            
            # Find available port if not specified
            if port is None:
                port = self._find_available_port()
            
            # Try to start the application
            if self._start_web_app(project_path, port):
                url = f"http://localhost:{port}"
                result['url'] = url
                
                # Wait for app to be ready
                time.sleep(5)
                
                # Capture screenshots
                if self.setup_driver():
                    try:
                        screenshots = self._capture_ui_screenshots(url)
                        result['screenshots'] = screenshots
                        result['success'] = True
                        
                        # Analyze the UI
                        result['analysis'] = self._analyze_ui(screenshots)
                    finally:
                        self.cleanup()
            else:
                result['errors'].append("Failed to start web application")
                
        except Exception as e:
            result['errors'].append(f"Execution error: {str(e)}")
        
        return result
    
    def _start_web_app(self, project_path: str, port: int) -> bool:
        """Start a web application based on project type"""
        try:
            original_dir = os.getcwd()
            os.chdir(project_path)
            
            # Check for package.json (Node.js/React/Vue/Angular)
            if Path("package.json").exists():
                result = self._start_node_app(port)
                os.chdir(original_dir)
                return result
            
            # Check for client-server structure (client/package.json)
            elif Path("client/package.json").exists():
                logger.info("Found client-server structure, starting from client directory")
                os.chdir("client")
                result = self._start_node_app(port)
                os.chdir(original_dir)
                return result
            
            # Check for requirements.txt (Python Flask/Django)
            elif Path("requirements.txt").exists():
                result = self._start_python_app(port)
                os.chdir(original_dir)
                return result
            
            # Check for index.html (Static HTML)
            elif Path("index.html").exists():
                result = self._start_static_app(port)
                os.chdir(original_dir)
                return result
            
            # Check for other frameworks
            elif Path("Cargo.toml").exists():
                result = self._start_rust_app(port)
                os.chdir(original_dir)
                return result
            
            elif Path("go.mod").exists():
                result = self._start_go_app(port)
                os.chdir(original_dir)
                return result
            
            else:
                print("Unknown project type - no web application files found")
                os.chdir(original_dir)
                return False
                
        except Exception as e:
            print(f"Error starting app: {e}")
            return False
    
    def _start_node_app(self, port: int) -> bool:
        """Start Node.js application using secure execution"""
        try:
            logger.info(f"Starting Node.js app on port {port}")
            
            # Check if npm is available
            npm_check = secure_executor.execute_safe(["npm", "--version"], timeout=10)
            if not npm_check['success']:
                logger.warning("npm not found - Node.js may not be installed")
                return False
            
            # Check if package.json exists
            if not Path("package.json").exists():
                logger.warning("No package.json found")
                return False
            
            # Install dependencies first
            install_result = secure_executor.execute_safe(
                ["npm", "install"], 
                timeout=config.security.max_timeout
            )
            
            if not install_result['success']:
                logger.warning(f"npm install failed: {install_result['stderr']}")
            else:
                logger.info("Dependencies installed successfully")
            
            # Try different start commands
            start_commands = [
                ["npm", "start"],
                ["npm", "run", "dev"],
                ["npm", "run", "serve"],
                ["npx", "serve", "-l", str(port)]
            ]
            
            for cmd in start_commands:
                try:
                    logger.info(f"Trying command: {' '.join(cmd)}")
                    
                    # Use secure execution with cleanup
                    result = secure_executor.execute_with_cleanup(
                        cmd, 
                        timeout=config.security.max_timeout
                    )
                    
                    if result['success']:
                        # Check if server is actually running
                        try:
                            response = requests.get(f"http://localhost:{port}", timeout=5)
                            if response.status_code == 200:
                                logger.info(f"Node.js app started successfully on port {port}")
                                return True
                        except requests.RequestException:
                            logger.warning(f"Server check failed for command: {' '.join(cmd)}")
                            continue
                    else:
                        logger.warning(f"Command failed: {result['stderr']}")
                        
                except Exception as e:
                    logger.error(f"Command execution error: {e}")
                    continue
            
            logger.error("All Node.js start commands failed")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Node.js app: {e}")
            raise UIExecutionError(f"Node.js app startup failed: {e}")
    
    def _start_python_app(self, port: int) -> bool:
        """Start Python application"""
        try:
            # Install dependencies
            subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
            
            # Try to find the main app file
            app_files = ["app.py", "main.py", "run.py", "server.py"]
            for app_file in app_files:
                if Path(app_file).exists():
                    process = subprocess.Popen([
                        "python", app_file
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(5)
                    return True
            
            return False
            
        except Exception as e:
            print(f"Failed to start Python app: {e}")
            return False
    
    def _start_static_app(self, port: int) -> bool:
        """Start static HTML application"""
        try:
            # Use Python's built-in server
            process = subprocess.Popen([
                "python", "-m", "http.server", str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
            
            # Check if server is actually running
            import requests
            try:
                response = requests.get(f"http://localhost:{port}", timeout=5)
                return response.status_code == 200
            except:
                return False
            
        except Exception as e:
            print(f"Failed to start static app: {e}")
            return False
    
    def _start_rust_app(self, port: int) -> bool:
        """Start Rust application"""
        try:
            # Build and run
            subprocess.run(["cargo", "build"], check=True)
            process = subprocess.Popen([
                "cargo", "run"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
            return True
            
        except Exception as e:
            print(f"Failed to start Rust app: {e}")
            return False
    
    def _start_go_app(self, port: int) -> bool:
        """Start Go application"""
        try:
            # Run the app
            process = subprocess.Popen([
                "go", "run", "."
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
            return True
            
        except Exception as e:
            print(f"Failed to start Go app: {e}")
            return False
    
    def _capture_ui_screenshots(self, url: str) -> List[Dict[str, Any]]:
        """Capture screenshots of the UI"""
        screenshots = []
        
        try:
            if not self.driver:
                raise UIExecutionError("WebDriver not initialized")
            
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Capture main page
            screenshot_path = self.screenshots_dir / "main_page.png"
            self.driver.save_screenshot(str(screenshot_path))
            
            screenshots.append({
                'name': 'main_page',
                'path': str(screenshot_path),
                'description': 'Main application page'
            })
            
            # Try to find and click interactive elements with improved error handling
            interactive_elements = self._find_interactive_elements()
            
            for i, element_info in enumerate(interactive_elements):
                try:
                    # Re-find the element to avoid stale reference
                    element = self._refind_element(element_info)
                    if not element:
                        continue
                    
                    # Click the element
                    element.click()
                    time.sleep(2)
                    
                    # Capture screenshot
                    screenshot_path = self.screenshots_dir / f"interaction_{i}.png"
                    if self.driver:
                        self.driver.save_screenshot(str(screenshot_path))
                    
                    screenshots.append({
                        'name': f'interaction_{i}',
                        'path': str(screenshot_path),
                        'description': f'After clicking {element.tag_name} element'
                    })
                    
                except Exception as e:
                    # Log the error but continue with other elements
                    logger.warning(f"Failed to interact with element {i}: {e}")
                    # Try to capture a screenshot even if interaction failed
                    try:
                        screenshot_path = self.screenshots_dir / f"error_{i}.png"
                        if self.driver:
                            self.driver.save_screenshot(str(screenshot_path))
                        screenshots.append({
                            'name': f'error_{i}',
                            'path': str(screenshot_path),
                            'description': f'Error state after failed interaction {i}'
                        })
                    except:
                        pass
                    continue
            
            # Try different routes/pages if it's a SPA
            self._explore_routes(screenshots)
            
        except Exception as e:
            print(f"Error capturing screenshots: {e}")
        
        return screenshots
    
    def _find_interactive_elements(self) -> List[Dict[str, Any]]:
        """Find interactive elements on the page and return element info"""
        interactive_selectors = [
            "button", "a", "input", "select", "textarea",
            "[onclick]", "[role='button']", ".btn", ".button"
        ]
        
        elements_info = []
        if not self.driver:
            return elements_info
            
        for selector in interactive_selectors:
            try:
                found_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in found_elements[:3]:  # Limit to first 3 per selector
                    try:
                        # Get element info for later re-finding
                        element_info = {
                            'selector': selector,
                            'tag_name': element.tag_name,
                            'text': element.text[:50] if element.text else '',  # Limit text length
                            'id': element.get_attribute('id') or '',
                            'class': element.get_attribute('class') or '',
                            'href': element.get_attribute('href') or '',
                            'onclick': element.get_attribute('onclick') or ''
                        }
                        elements_info.append(element_info)
                    except Exception as e:
                        logger.warning(f"Error getting element info: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error finding elements with selector {selector}: {e}")
                continue
        
        return elements_info[:5]  # Limit total interactions
    
    def _refind_element(self, element_info: Dict[str, Any]):
        """Re-find an element using its stored information to avoid stale references"""
        try:
            # Try to find by ID first (most reliable)
            if element_info.get('id'):
                try:
                    return self.driver.find_element(By.ID, element_info['id'])
                except:
                    pass
            
            # Try to find by class and tag
            if element_info.get('class') and element_info.get('tag_name'):
                try:
                    class_selector = f"{element_info['tag_name']}.{element_info['class'].replace(' ', '.')}"
                    return self.driver.find_element(By.CSS_SELECTOR, class_selector)
                except:
                    pass
            
            # Try to find by href (for links)
            if element_info.get('href'):
                try:
                    return self.driver.find_element(By.CSS_SELECTOR, f"a[href='{element_info['href']}']")
                except:
                    pass
            
            # Fallback to text content (less reliable)
            if element_info.get('text'):
                try:
                    return self.driver.find_element(By.LINK_TEXT, element_info['text'])
                except:
                    pass
            
            # Last resort: try original selector
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, element_info['selector'])
                if elements:
                    return elements[0]  # Return first matching element
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Error re-finding element: {e}")
            return None
    
    def _explore_routes(self, screenshots: List[Dict[str, Any]]):
        """Explore different routes in a SPA"""
        try:
            if not self.driver:
                return
                
            # Try common routes
            routes = ["/about", "/contact", "/dashboard", "/profile", "/settings"]
            
            for route in routes:
                try:
                    current_url = self.driver.current_url
                    self.driver.get(f"{current_url.rstrip('/')}{route}")
                    time.sleep(2)
                    
                    screenshot_path = self.screenshots_dir / f"route_{route.replace('/', '_')}.png"
                    self.driver.save_screenshot(str(screenshot_path))
                    
                    screenshots.append({
                        'name': f'route_{route.replace("/", "_")}',
                        'path': str(screenshot_path),
                        'description': f'Route: {route}'
                    })
                    
                except:
                    continue
                    
        except Exception as e:
            print(f"Error exploring routes: {e}")
    
    def _analyze_ui(self, screenshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze captured UI screenshots"""
        analysis = {
            'visual_quality': 0,
            'accessibility': 0,
            'responsiveness': 0,
            'interactivity': 0,
            'issues': [],
            'recommendations': []
        }
        
        for screenshot in screenshots:
            try:
                # Load and analyze the image
                img = cv2.imread(screenshot['path'])
                if img is not None:
                    img_analysis = self._analyze_image(img)
                    
                    # Combine analysis results
                    analysis['visual_quality'] += img_analysis.get('visual_quality', 0)
                    analysis['accessibility'] += img_analysis.get('accessibility', 0)
                    analysis['responsiveness'] += img_analysis.get('responsiveness', 0)
                    analysis['interactivity'] += img_analysis.get('interactivity', 0)
                    
                    analysis['issues'].extend(img_analysis.get('issues', []))
                    analysis['recommendations'].extend(img_analysis.get('recommendations', []))
                    
            except Exception as e:
                print(f"Error analyzing screenshot {screenshot['path']}: {e}")
        
        # Average the scores
        num_screenshots = len(screenshots)
        if num_screenshots > 0:
            analysis['visual_quality'] /= num_screenshots
            analysis['accessibility'] /= num_screenshots
            analysis['responsiveness'] /= num_screenshots
            analysis['interactivity'] /= num_screenshots
        
        return analysis
    
    def _analyze_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze a single image for UI quality"""
        analysis = {
            'visual_quality': 0,
            'accessibility': 0,
            'responsiveness': 0,
            'interactivity': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Analyze visual quality
            analysis['visual_quality'] = self._assess_visual_quality(img, gray)
            
            # Analyze accessibility
            analysis['accessibility'] = self._assess_accessibility(img, gray)
            
            # Analyze responsiveness
            analysis['responsiveness'] = self._assess_responsiveness(img)
            
            # Analyze interactivity
            analysis['interactivity'] = self._assess_interactivity(img, gray)
            
        except Exception as e:
            print(f"Error in image analysis: {e}")
        
        return analysis
    
    def _assess_visual_quality(self, img: np.ndarray, gray: np.ndarray) -> float:
        """Assess visual quality of the UI"""
        try:
            # Calculate image sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate color diversity
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            color_diversity = min(unique_colors / 1000, 1.0)
            
            # Calculate contrast
            contrast = gray.std()
            contrast_score = min(contrast / 100, 1.0)
            
            # Combine metrics
            quality_score = (laplacian_var / 1000 + color_diversity + contrast_score) / 3
            return min(quality_score, 10.0)
            
        except:
            return 5.0
    
    def _assess_accessibility(self, img: np.ndarray, gray: np.ndarray) -> float:
        """Assess accessibility features"""
        try:
            # Check for high contrast
            contrast_score = self._check_contrast(img)
            
            # Check for text readability
            text_score = self._check_text_readability(gray)
            
            # Check for color accessibility
            color_score = self._check_color_accessibility(img)
            
            accessibility_score = (contrast_score + text_score + color_score) / 3
            return min(accessibility_score * 10, 10.0)
            
        except:
            return 5.0
    
    def _check_contrast(self, img: np.ndarray) -> float:
        """Check contrast ratio"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_intensity = gray.mean()
            std_intensity = gray.std()
            
            # Higher std means better contrast
            contrast_ratio = std_intensity / 100
            return min(contrast_ratio, 1.0)
        except:
            return 0.5
    
    def _check_text_readability(self, gray: np.ndarray) -> float:
        """Check text readability"""
        try:
            # Use edge detection to find text areas
            edges = cv2.Canny(gray, 50, 150)
            text_areas = int(np.sum(edges.astype(bool)))
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # More edges might indicate more text
            text_ratio = text_areas / total_pixels
            return float(min(text_ratio * 10, 1.0))
        except:
            return 0.5
    
    def _check_color_accessibility(self, img: np.ndarray) -> float:
        """Check color accessibility"""
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Check for color diversity (not just grayscale)
            unique_hues = len(np.unique(hsv[:, :, 0]))
            color_diversity = min(unique_hues / 50, 1.0)
            
            return color_diversity
        except:
            return 0.5
    
    def _assess_responsiveness(self, img: np.ndarray) -> float:
        """Assess responsiveness indicators"""
        try:
            height, width = img.shape[:2]
            aspect_ratio = width / height
            
            # Check if it's a reasonable aspect ratio
            if 0.5 <= aspect_ratio <= 2.0:
                return 8.0
            elif 0.3 <= aspect_ratio <= 3.0:
                return 6.0
            else:
                return 4.0
        except:
            return 5.0
    
    def _assess_interactivity(self, img: np.ndarray, gray: np.ndarray) -> float:
        """Assess interactivity indicators"""
        try:
            # Look for buttons, links, and interactive elements
            # Use edge detection to find rectangular shapes (buttons)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count rectangular shapes (potential buttons)
            button_count = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangle
                    button_count += 1
            
            # More buttons/interactive elements = higher score
            interactivity_score = min(button_count / 5, 1.0) * 10
            return interactivity_score
            
        except:
            return 5.0
    
    def test_functionality(self, url: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test specific functionality of the application"""
        results = {
            'passed': 0,
            'failed': 0,
            'total': len(test_cases),
            'details': []
        }
        
        if not self.setup_driver():
            return results
        
        try:
            if not self.driver:
                raise UIExecutionError("WebDriver not initialized")
                
            self.driver.get(url)
            time.sleep(3)
            
            for i, test_case in enumerate(test_cases):
                try:
                    result = self._execute_test_case(test_case)
                    results['details'].append(result)
                    
                    if result['passed']:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    results['details'].append({
                        'name': test_case.get('name', f'Test {i}'),
                        'passed': False,
                        'error': str(e)
                    })
                    results['failed'] += 1
                    
        finally:
            self.cleanup()
        
        return results
    
    def _execute_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test case"""
        result = {
            'name': test_case.get('name', 'Unknown test'),
            'passed': False,
            'error': None
        }
        
        if not self.driver:
            result['error'] = "WebDriver not initialized"
            return result
        
        try:
            action = test_case.get('action')
            selector = test_case.get('selector')
            expected = test_case.get('expected')
            
            if action == 'click':
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                element.click()
                time.sleep(1)
                result['passed'] = True
                
            elif action == 'type':
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                element.clear()
                element.send_keys(test_case.get('text', ''))
                time.sleep(1)
                result['passed'] = True
                
            elif action == 'check_text':
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                actual_text = element.text
                result['passed'] = actual_text == expected
                result['actual'] = actual_text
                result['expected'] = expected
                
            elif action == 'check_url':
                actual_url = self.driver.current_url
                if actual_url and expected:
                    result['passed'] = expected in str(actual_url)
                else:
                    result['passed'] = False
                result['actual'] = actual_url
                result['expected'] = expected
                
        except Exception as e:
            result['error'] = str(e)
        
        return result


class UIExecutionAgent:
    """Enhanced agent that can execute and analyze UIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.renderer = UIRenderer()
    
    def analyze_with_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a project with UI execution capabilities"""
        analysis = {
            'execution_success': False,
            'ui_analysis': {},
            'screenshots': [],
            'functionality_tests': {},
            'recommendations': []
        }
        
        try:
            # Get project path
            project_path = context.get('project_path', '.')
            
            # Try to execute the application
            execution_result = self.renderer.execute_web_app(project_path)
            
            if execution_result['success']:
                analysis['execution_success'] = True
                analysis['ui_analysis'] = execution_result['analysis']
                analysis['screenshots'] = execution_result['screenshots']
                
                # Run functionality tests
                if execution_result['url']:
                    test_cases = self._generate_test_cases(context)
                    analysis['functionality_tests'] = self.renderer.test_functionality(
                        execution_result['url'], test_cases
                    )
                
                # Generate recommendations
                analysis['recommendations'] = self._generate_recommendations(analysis)
            
            else:
                analysis['errors'] = execution_result['errors']
                
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_test_cases(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases based on project type"""
        test_cases = [
            {
                'name': 'Page loads successfully',
                'action': 'check_url',
                'selector': 'body',
                'expected': 'localhost'
            },
            {
                'name': 'Has interactive elements',
                'action': 'click',
                'selector': 'button, a, input',
                'expected': None
            }
        ]
        
        # Add framework-specific tests
        if 'react' in context.get('technologies', []):
            test_cases.extend([
                {
                    'name': 'React app renders',
                    'action': 'check_text',
                    'selector': '[data-testid], .App',
                    'expected': None
                }
            ])
        
        if 'vue' in context.get('technologies', []):
            test_cases.extend([
                {
                    'name': 'Vue app renders',
                    'action': 'check_text',
                    'selector': '#app',
                    'expected': None
                }
            ])
        
        return test_cases
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on UI analysis"""
        recommendations = []
        
        ui_analysis = analysis.get('ui_analysis', {})
        
        if ui_analysis.get('visual_quality', 0) < 6:
            recommendations.append("Improve visual design and layout")
        
        if ui_analysis.get('accessibility', 0) < 6:
            recommendations.append("Enhance accessibility features")
        
        if ui_analysis.get('interactivity', 0) < 6:
            recommendations.append("Add more interactive elements")
        
        if ui_analysis.get('responsiveness', 0) < 6:
            recommendations.append("Improve responsive design")
        
        return recommendations

