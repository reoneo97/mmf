
Classification
- Get data from the encoder
- Pool the output from the text and the visual
- Pass through classifier head 

How to get feature variables from the raw images 



VILBERT

- Text Embedding `transformers.bert.BertEmbeddings` -
- Vision Embedding - `BertImageFeatureEmbedding`

- Encoder


    - Takes in embedding output and the masks
- Text and Visual Pooler

-   Convert the N X L x D output to a smaller output based on the first token 


## ViLBert Encoder
- Contains BertLayer, BertImage Layer and BertConnectionLayers

### Forward Function
- Takes in embedding for tokens and images and spits output 

## VILBERT Image Layer `BertImageLayer`
- Contains:
    - BertImageAttention
        - The BERT Image attention takes in input_tensor, attention_mask, txt_embedding and txt_attention_mask
        - Contains: BertImageSelfAttention and BertImageSelfOutput
    - BertImageIntermediate
    - BertImageOutput
## VILBERT Text Layer
- Contains

How to connect the layers by attention

Image Extractors
google/vit-large-patch16-224-in21k
google/vit-base-patch16-224-in21k
google/vit-huge-patch16-224-in21k

Need to print out the embeddings size to see how to configure the additional encoder portion

Does pre-training on alignment help with being able to detect hateful speech 