### Setup Instructions
  
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Mount Google Drive and set data path in the notebook

3. Run all cells in the notebook sequentially

### Reproduction Instructions

1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Upload the dataset to your Drive and update the `DATA_PATH` variable
4. Run all cells from top to bottom
5. Models and results will be saved to your Drive

### File Structure
```
├── 62FIT4ATI_Group_X_Topic_5.ipynb    # Main notebook
├── best_model_t5.pt                    # Best model checkpoint
├── t5_title_generation_final.pt        # Final model
├── t5_tokenizer/                       # Saved tokenizer
├── evaluation_results.json             # Metrics and results
├── test_predictions.csv                # Test set predictions
├── training_history.png                # Training curves
├── performance_analysis.png            # Performance visualization
└── README.md                           # This file
```

### Usage Example

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('path/to/model')
tokenizer = T5Tokenizer.from_pretrained('path/to/tokenizer')

# Generate title
abstract = "Your scientific abstract here..."
input_text = "summarize: " + abstract
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

outputs = model.generate(
    inputs['input_ids'],
    max_length=64,
    num_beams=5,
    early_stopping=True
)

title = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Title: {title}")
```

### Key Findings

1. **Optimization Impact**: 
   - Learning rate warmup reduced initial training instability
   - Gradient accumulation enabled effective batch size of 32
   
2. **Performance Analysis**:
   - Model achieves 34.6% excellent predictions (ROUGE-L ≥ 0.5)
   - Average title length closely matches ground truth

3. **Challenges Addressed**:
   - Handled variable-length inputs with proper tokenization
   - Addressed GPU memory constraints with gradient accumulation
   - Prevented overfitting with early stopping

### Future Improvements
- Experiment with larger T5 variants (t5-large, t5-3b)
- Implement domain-specific fine-tuning for different scientific fields
- Add diversity-promoting decoding strategies
- Explore prompt engineering techniques

### References
1. Raffel et al. (2020). Exploring the Limits of Transfer Learning with T5
2. Lewis et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training
3. Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries

### Contact
For questions or issues, please contact [your email]

### License
This project is for educational purposes as part of 62FIT4ATI course.
