from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_summary(text, max_length=150, min_length=100):
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs,
                                max_length=max_length,
                                min_length=min_length, 
                                num_beams=4, 
                                length_penalty=2.0, 
                                top_k=50, 
                                top_p=0.95, 
                                early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary