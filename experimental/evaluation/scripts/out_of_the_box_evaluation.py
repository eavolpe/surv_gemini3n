import json
import re
import os
import base64
import requests
import fire
import time
from pathlib import Path
from PIL import Image, ImageEnhance
import io
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

class SecurityModelEvaluator:
    def __init__(self, dataset_path: str = "/Users/rafael/Factored/surv_gemini3n/experimental/datasets/data/anomaly_frames"):
        self.dataset_path = dataset_path
        self.models = ['gsec2b', 'gsec4b']
        self.ollama_url = 'http://localhost:11434/api/generate'
        self.results = []
        
    
    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 encoding for API request"""
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((512,512))
            #enhancer = ImageEnhance.Contrast(img)
            #img = enhancer.enhance(1.5)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def query_model(self, image_path: str, model: str) -> Optional[Dict]:
        """Send image to model and get structured response"""
        b64_img = self.encode_image(image_path)
        if not b64_img:
            return None
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2)  
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        'model': model,
                        'prompt': 'Analyze this surveillance image for security threats and anomalies.',
                        'images': [b64_img],
                        'stream': False,
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get('response', '')
                    
                    if any(phrase in response_text.lower() for phrase in [
                        'unable to view images', 
                        'unable to process images',
                        'cannot analyze the surveillance image',
                        'i am a text-based ai'
                    ]):
                        print(f"Model {model} gave 'unable to view' response on attempt {attempt + 1}, retrying...")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            print(f"Model {model} consistently unable to process images for {image_path}")
                            return None
                    
                    return response_data
                else:
                    print(f"API error for {model} on {image_path}: {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"Request error for {model} on {image_path}: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def parse_model_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from model response"""
        if not response_text:
            return None
            
        try:
            markdown_json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            markdown_match = re.search(markdown_json_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if markdown_match:
                json_str = markdown_match.group(1)
                try:
                    parsed = json.loads(json_str)
                    required_fields = ['classification', 'confidence', 'description', 'risk_level']
                    if all(field in parsed for field in required_fields):
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                required_fields = ['classification', 'confidence', 'description', 'risk_level']
                if all(field in parsed for field in required_fields):
                    return parsed
            
            print(f"Failed to parse JSON, attempting manual extraction from: {response_text[:200]}...")
            return self.manual_parse_response(response_text)
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return self.manual_parse_response(response_text)
    
    def manual_parse_response(self, response_text: str) -> Optional[Dict]:
        """Fallback manual parsing for non-JSON responses"""
        # Extract classification from common patterns
        classification_patterns = [
            r'classification["\s]*:?["\s]*([A-Za-z]+)',
            r'Incident Type["\s]*:?["\s]*([A-Za-z]+)',
            r'detected["\s]*:?["\s]*([A-Za-z]+)',
        ]
        
        classification = None
        for pattern in classification_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                classification = match.group(1)
                break
        
        if not classification:
            if re.search(r'no danger|normal|nothing suspicious', response_text, re.IGNORECASE):
                classification = "Normal"
            else:
                classification = "Unknown"
        
        return {
            "classification": classification,
            "confidence": "unknown",
            "description": response_text[:100] + "..." if len(response_text) > 100 else response_text,
            "risk_level": "unknown"
        }
    
    def get_dataset_structure(self) -> Dict[str, List[str]]:
        """Get all categories and their images"""
        dataset_structure = {}
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path not found: {self.dataset_path}")
            return dataset_structure
        
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if os.path.isdir(category_path):
                images = [f for f in os.listdir(category_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:  
                    dataset_structure[category] = sorted(images)
                    
        return dataset_structure
    
    def evaluate_subset(self, max_images_per_category: int = 3):
        """Evaluate a small subset for testing"""
        print(f"Running subset evaluation ({max_images_per_category} images per category)")
        dataset_structure = self.get_dataset_structure()
        
        total_images = 0
        for category, images in dataset_structure.items():
            subset_images = images[:max_images_per_category]
            total_images += len(subset_images)
            
            for image_file in subset_images:
                image_path = os.path.join(self.dataset_path, category, image_file)
                self.evaluate_single_image(image_path, category, image_file)
        
        print(f"Completed subset evaluation: {total_images} images")
        return self.results
    
    def evaluate_full(self):
        """Evaluate complete dataset"""
        print("Running full dataset evaluation")
        dataset_structure = self.get_dataset_structure()
        
        total_images = sum(len(images) for images in dataset_structure.values())
        processed = 0
        
        for category, images in dataset_structure.items():
            print(f"Processing category: {category} ({len(images)} images)")
            
            for image_file in images:
                image_path = os.path.join(self.dataset_path, category, image_file)
                self.evaluate_single_image(image_path, category, image_file)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Progress: {processed}/{total_images} images processed")
        
        print(f"Completed full evaluation: {total_images} images")
        return self.results
    
    def evaluate_single_image(self, image_path: str, ground_truth: str, image_file: str):
        """Evaluate single image with both models"""
        print(f"Evaluating: {ground_truth}/{image_file}")
        
        for i, model in enumerate(self.models):
            if i > 0:
                time.sleep(1)
            api_response = self.query_model(image_path, model)
            if not api_response:
                continue
                
            parsed_response = self.parse_model_response(api_response.get('response', ''))
            if not parsed_response:
                continue
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "image_file": image_file,
                "ground_truth": ground_truth,
                "model": model,
                "prediction": parsed_response,
                "correct": parsed_response.get('classification', '').lower() == ground_truth.lower(),
                "raw_response": api_response.get('response', '')
            }
            
            self.results.append(result)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        metrics = {}
        
        for model in self.models:
            model_results = [r for r in self.results if r['model'] == model]
            if model_results:
                correct = sum(1 for r in model_results if r['correct'])
                total = len(model_results)
                metrics[f"{model}_accuracy"] = correct / total
                metrics[f"{model}_total_images"] = total
        
        category_metrics = defaultdict(lambda: defaultdict(dict))
        for result in self.results:
            category = result['ground_truth']
            model = result['model']
            
            if category not in category_metrics:
                category_metrics[category] = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            category_metrics[category][model]['total'] += 1
            if result['correct']:
                category_metrics[category][model]['correct'] += 1
        
        for category, model_data in category_metrics.items():
            for model, stats in model_data.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                metrics[f"{category}_{model}_accuracy"] = accuracy
                metrics[f"{category}_{model}_count"] = stats['total']
        
        confusion_data = defaultdict(lambda: defaultdict(int))
        for result in self.results:
            predicted = result['prediction'].get('classification', 'Unknown')
            actual = result['ground_truth']
            confusion_data[actual][predicted] += 1
        
        metrics['confusion_matrix'] = dict(confusion_data)
        
        return metrics
    
    def save_results(self, output_file: str = None):
        """Save evaluation results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.json"
        
        metrics = self.calculate_metrics()
        
        output_data = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "dataset_path": self.dataset_path,
                "models": self.models,
                "total_results": len(self.results)
            },
            "metrics": metrics,
            "detailed_results": self.results
        }
        
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        
        self.print_summary(metrics)
        
        return output_path
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for model in self.models:
            acc_key = f"{model}_accuracy"
            count_key = f"{model}_total_images"
            if acc_key in metrics:
                accuracy = metrics[acc_key] * 100
                count = metrics[count_key]
                print(f"{model.upper()}: {accuracy:.1f}% accuracy ({count} images)")
        
        print("\n" + "-"*30)
        print("PER-CATEGORY PERFORMANCE")
        print("-"*30)
        
        categories = set()
        for key in metrics.keys():
            if key.endswith('_accuracy') and not key.startswith(('gsec2b', 'gsec4b')):
                category = key.replace('_gsec2b_accuracy', '').replace('_gsec4b_accuracy', '')
                categories.add(category)
        
        for category in sorted(categories):
            print(f"\n{category}:")
            for model in self.models:
                acc_key = f"{category}_{model}_accuracy"
                count_key = f"{category}_{model}_count"
                if acc_key in metrics:
                    accuracy = metrics[acc_key] * 100
                    count = metrics[count_key]
                    print(f"  {model}: {accuracy:.1f}% ({count} images)")

def test_subset(max_images: int = 3):
    """Run evaluation on subset of data"""
    evaluator = SecurityModelEvaluator()
    evaluator.evaluate_subset(max_images)
    return evaluator.save_results("subset_evaluation.json")

def full_evaluation():
    """Run evaluation on complete dataset"""
    evaluator = SecurityModelEvaluator()
    evaluator.evaluate_full()
    return evaluator.save_results("full_evaluation.json")

def analyze_results(results_file: str):
    """Analyze existing results file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    evaluator = SecurityModelEvaluator()
    evaluator.results = data['detailed_results']
    metrics = evaluator.calculate_metrics()
    evaluator.print_summary(metrics)

if __name__ == '__main__':
    fire.Fire({
        'test': test_subset,
        'full': full_evaluation,
        'analyze': analyze_results
    })