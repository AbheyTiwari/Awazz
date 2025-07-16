import os
import numpy as np

def test_combined_model():
    """Test the combined SOS model if it exists"""
    model_path = 'combined_sos_model.tflite'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run 'py combine_models.py' first to create the combined model")
        return False
    
    try:
        import tensorflow as tf
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Combined Model Test Results")
        print("=" * 40)
        print(f"Model: {model_path}")
        print(f"Size: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        print(f"\nInputs ({len(input_details)}):")
        for i, detail in enumerate(input_details):
            print(f"  {i+1}. {detail['name']}: {detail['shape']}")
        
        print(f"\nOutputs ({len(output_details)}):")
        for i, detail in enumerate(output_details):
            print(f"  {i+1}. {detail['name']}: {detail['shape']}")
        
        # Test with dummy data
        print("\nTesting with dummy data...")
        
        for i, detail in enumerate(input_details):
            dummy_input = np.random.random(detail['shape']).astype(detail['dtype'])
            interpreter.set_tensor(detail['index'], dummy_input)
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        score = float(output[0][0])  # Extract the actual score value
        
        print(f"\nEmergency Score: {score:.3f}")
        
        if score > 0.7:
            print("🚨 HIGH EMERGENCY ALERT!")
        elif score > 0.4:
            print("⚠️  Moderate emergency detected")
        else:
            print("✅ Normal situation")
        
        return True
        
    except ImportError:
        print("TensorFlow not installed. Please install using:")
        print("pip install tensorflow")
        return False
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

def main():
    print("Emergency SOS Model Tester")
    print("=" * 40)
    
    # Check if original models exist
    models = [
        'keyword_model_quant.tflite',
        'quantized_emotion_recognition_model.tflite',
        'screamdetector_model.tflite'
    ]
    
    print("Checking original models:")
    for model in models:
        if os.path.exists(model):
            size = os.path.getsize(model) / 1024
            print(f"✓ {model}: {size:.2f} KB")
        else:
            print(f"✗ {model}: Not found")
    
    print("\nChecking combined model:")
    test_combined_model()

if __name__ == "__main__":
    main()
