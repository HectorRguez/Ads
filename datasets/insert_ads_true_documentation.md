# TRUE Benchmark Dataset Question Generation Strategy

## Overview
Each TRUE benchmark dataset represents a different text generation task. To fairly evaluate your model's factual consistency, we need to replicate the original generation task that produced the text being evaluated. This document justifies the question format for each dataset.

## Dataset-Specific Question Generation

### 1. **SummEval** - News Article Summarization
**Original Task**: Generate summaries of CNN/DailyMail news articles
**Question Format**: `"Please write a concise summary of the following news article:\n\n{article}"`

**Justification**:
- SummEval evaluates summarization quality, so we need your model to generate summaries
- The original generated texts in SummEval are summaries of news articles
- By asking for summarization, we replicate the exact task that created the original text
- This allows fair comparison of factual consistency between your summaries and the benchmark summaries

### 2. **QAGS-CNNDM** - Question-Answering Based Summarization
**Original Task**: Answer questions about news articles or provide key information
**Question Format**: `"Based on the following news article, please provide the key information and main points:\n\n{article}"`

**Justification**:
- QAGS (Question Answering for Generating Summaries) evaluates factual consistency in summarization
- The original task involves extracting key information from articles
- We ask for "key information and main points" to mirror the information extraction task
- This generates text that can be fairly compared for factual accuracy

### 3. **QAGS-XSum** - Extreme Summarization
**Original Task**: Generate one-sentence summaries of BBC articles
**Question Format**: `"Please provide a one-sentence summary that captures the essence of this BBC article:\n\n{article}"`

**Justification**:
- XSum dataset requires extremely concise, one-sentence summaries
- The constraint of one sentence forces models to prioritize the most important information
- This replicates the compression challenge of the original task
- Factual consistency is crucial when information must be heavily condensed

### 4. **PAWS** - Paraphrase Adversaries from Word Scrambling
**Original Task**: Determine semantic equivalence between sentence pairs
**Question Format**: `"Please rephrase the following sentence while preserving its exact meaning:\n\n{sentence}"`

**Justification**:
- PAWS evaluates whether models can distinguish between paraphrases and near-paraphrases
- The original task involves semantic similarity judgment
- By asking for paraphrasing, we test your model's ability to preserve meaning while changing form
- This directly tests factual/semantic consistency preservation

### 5. **FEVER** - Fact Extraction and Verification
**Original Task**: Classify claims as SUPPORTED/REFUTED/NOT_ENOUGH_INFO given evidence
**Question Format**: `"Based on the following evidence, please explain what conclusions can be drawn and provide a factual analysis:\n\nEvidence: {evidence}"`

**Justification**:
- FEVER is explicitly about fact verification using evidence
- The original task requires reasoning about evidence to verify claims
- We ask for "conclusions and factual analysis" to mirror the reasoning process
- This tests your model's ability to make factually grounded inferences

### 6. **DialFact** - Dialogue Factual Consistency
**Original Task**: Generate factually consistent responses in dialogue
**Question Format**: `"Please continue this conversation with a factually accurate and helpful response:\n\n{dialogue_context}"`

**Justification**:
- DialFact evaluates factual consistency in conversational AI
- The original task involves generating dialogue responses
- We explicitly request "factually accurate" responses to emphasize the consistency requirement
- This replicates the conversational generation task while emphasizing factual grounding

### 7. **Q²** - Question Generation and Answering
**Original Task**: Generate questions and answers from context
**Question Format**: `"Based on the following context, please provide relevant questions and their answers:\n\n{context}"`

**Justification**:
- Q² evaluates question-answer generation from context
- The original task involves creating Q&A pairs
- We ask for both questions and answers to mirror the dual generation task
- This tests factual consistency in both question formulation and answer generation

### 8. **FRANK** - Factual Consistency in Text Rewriting
**Original Task**: Rewrite text while maintaining factual accuracy
**Question Format**: `"Please rewrite the following text to make it clearer and more accessible while maintaining all factual information:\n\n{text}"`

**Justification**:
- FRANK specifically evaluates factual consistency during text rewriting
- The original task involves text simplification/rewriting
- We explicitly mention "maintaining all factual information" to emphasize the consistency requirement
- This directly tests the model's ability to preserve facts during text transformation

### 9. **BEGIN** - Benchmark for Text Generation Evaluation
**Original Task**: Generate text continuations or expansions
**Question Format**: `"Please continue or expand on the following text in a natural and informative way:\n\n{text}"`

**Justification**:
- BEGIN evaluates general text generation capabilities
- The original task involves text continuation or expansion
- We ask for "natural and informative" text to encourage factual accuracy
- This tests consistency when generating additional content based on existing text

### 10. **MNBM** - Multi-Reference Benchmark
**Original Task**: Generate text with multiple valid reference outputs
**Question Format**: `"Please provide an informative response based on the following context:\n\n{context}"`

**Justification**:
- MNBM evaluates generation quality with multiple valid references
- The original task is more open-ended generation
- We ask for "informative response" to encourage factual content
- This tests consistency in open-domain generation tasks

### 11. **VitaminC** - Fact Verification with Evidence
**Original Task**: Verify claims against evidence passages using fine-grained annotations
**Question Format**: `"Based on the following evidence, please verify and explain whether the associated claims are factually supported:\n\nEvidence: {evidence}"`

**Justification**:
- VitaminC evaluates fact verification with detailed claim-evidence relationships
- The original task involves reasoning about evidence to verify specific claims
- We ask for verification and explanation to mirror the reasoning process
- This tests your model's ability to make precise factual judgments based on evidence
- The dataset includes fine-grained annotations making it ideal for factual consistency evaluation

## Fallback Strategy

When dataset type cannot be determined, we analyze the relationship between grounding text and generated text:

1. **Length Ratio < 0.3**: Likely summarization → Ask for summary
2. **Contains Questions**: Likely Q&A → Ask for insights/answers  
3. **Similar Length**: Likely continuation → Ask for expansion

## Key Principles

1. **Task Fidelity**: Each question replicates the original generation task
2. **Factual Emphasis**: Questions emphasize accuracy and consistency where relevant
3. **Context Preservation**: All questions include the full grounding context
4. **Length Management**: Context is truncated to 800 characters to avoid token limits
5. **Specificity**: Avoid generic "provide relevant response" prompts

## Expected Outcomes

By using task-specific questions:
- Your model generates text comparable to the original benchmark text
- Factual consistency evaluation is fair and meaningful
- Results reflect your model's performance on the actual tasks these benchmarks evaluate
- Comparison between with/without ads is valid since the generation task is consistent

This approach ensures that the TRUE benchmark evaluation accurately reflects your model's factual consistency across diverse text generation scenarios.