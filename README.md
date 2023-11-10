# NLP_similar_interests
NLP assignment on finding MCDA classmates with similar interests

## What are Embeddings?
Word and sentence embeddings are a clever way of turning words and sentences into numerical representations that computers can understand. Word and sentence embeddings make it possible for computers to work with and analyze text, making tasks like language translation, sentiment analysis, and information retrieval much easier and more accurate. A common way to numerically represent words and sentences is to transform them into vectors. Vectors are essentially a bunch of numbers arranged in a specific order, and they represent different aspects of the word or sentence.

For word embeddings, imagine each word as a unique vector. These vectors capture the word's meaning in a multi-dimensional space. Each dimension might represent something like word frequency, context, or similarity to other words. So, when we take a word and transform it into a vector, we're assigning it a specific location in this multi-dimensional space, sort of like plotting points on a map.
![image](https://github.com/htao00/NLP_similar_interests/assets/16727807/f8b2c191-501b-4e17-be8d-96457403a32a)
We expect related words or words with similar meaning to be used in similar contexts and appear with similar number of frequencies. Therefore, once tranformed into vectors we should see the similarities reflected in vector space. As shown in the plot above [1], related words are clustered close with each other.

Sentence embeddings are like puzzles made up of word vectors. When we form a sentence, we combine the individual word vectors in a specific way to create a new vector that represents the entire sentence. This combined vector captures the overall meaning and context of the sentence by taking into account the relationships between the words.

![image](https://github.com/htao00/NLP_similar_interests/assets/16727807/68e6f07b-6f1b-40ce-b389-798a30fd03bc)

In the image above[2], SIF (smooth inverse frequency)[3] refers to a sentence embedding method that combines word embeddings to calculate the sentence embedding.

The closer the vectors are, the more similar the words or sentences are in meaning. So, word and sentence embeddings are like a bridge that helps computers make sense of our words and language.

## Data Analysis
In this experiment, we are testing how sentence embeddings are compared in Sentence Transformers. We've modified sentences from 3 people in our dataset by changing the word orders in a sentence, replacing words with synonyms or antonyms, shortening the sentence, writing the oppsoite sentence meaning, or replacing the whole sentence with a randomly generated sentence without context. The changes are as follows:
<img width="957" alt="Screenshot 2023-11-09 at 23 13 34" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/f0db454f-02fb-4473-b1bf-87f90da82ef8">
<img width="957" alt="Screenshot 2023-11-09 at 23 14 05" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/a4d5a92b-cea1-4582-aef6-e3e11eb5624e">

We ran the matchmaking visualization code again and got the following:
![visualization_modified](https://github.com/htao00/NLP_similar_interests/assets/16727807/f249adf6-c88b-46de-a68f-3210ec6cdbd6)

As shown, most of Nikita's sentence modifications are still relatively clustered close together despite the changes (with the exception of Nikita_opposite2). While by replacing with a completely random sentence, Greg's and Tao's modified Tao's modified sentences are placed further away from their originals. To understand further why this is, we've calculated cosine similarities between the original sentence with modiefied ones and the results from the test are as follows for Nikita's sentence modifications:

<img width="771" alt="Screenshot 2023-11-09 at 22 39 16" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/290cfe39-1276-42c8-b779-b52ec130cb85">

And for Greg and Tao's:

<img width="790" alt="Screenshot 2023-11-09 at 23 22 53" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/d322da65-5185-4ab8-bcf2-6b0cae3c3cc0">

The results from the cosine similarity test largely corresponds with their placements on the plot above. Expectedly, replacing whole sentence have the highest impact on similarity with both Greg and Tao's modifications scoring near 0 on cosine similarity scale; replacing with synonyms had the least impact with all of Nikita's synonym modifications scoring near the 0.9 mark. Interestingly, we see that using antonyms does not have a large impact on the sentence similarities, yet, the sentences have the opposite meaning. Therefore, using this model, people with opposite interests would be placed close together. Note that while Nikita_opposite2 has a higher similarity score than Tao_mod and Greg_mod, it is placed further away from its original on the visualization plot. This is likely more to do with UMAP's dimensionality reduction mechanism. In general, on the plot, sentences with similar meaning is placed closer together than sentences that are not.

## Embedding Sensitivity Tests
In this expriment, we are testing how different transformer models affect our outcome of finding people with similar interests as me (Haodong Tao). We calculate the embedding similarities of 4 other transformer models (all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2, paraphrase-albert-small-v2) and calculate the spearman's correlation with our base model (all-MiniLM-L6-v2). We also rank order the people with closest descriptions to mine for all models and calculate the rank differences. The results are as follows.

![model_comparisons](https://github.com/htao00/NLP_similar_interests/assets/16727807/5e23166b-1a30-46c7-89f4-a4e06706558e)

As shown, all models have high correlation with our base model, with all-mpnet-base-v2 performs most differently from our base model with correlation of 0.728. This suggests that model embedding could be a significant factor in our task performance. We can demonstrate this further with the following analysis.

<img width="823" alt="Screenshot 2023-11-10 at 13 35 36" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/5fe8c45c-92af-4e8c-8650-bda9f115dd58">

Here we rank order people with the most similar interests as me (Haodong Tao) based on their description's cosine similarity score and calculate the difference in ranking. Then we count the number of people ranked 10 places or greater apart when comparing two embedding models (this difference is enough to displace the number 1 position out of top 10), therefore the higher this count is, the greater the difference in model performance. We also count the number of people ranked within 5 places apart between two models (this means a top 1 position would still be within the top 5), therefore, the higher this count is, the more similar in model performance. As we can see, the counts corresponds to the models' spearman correlations.

We can also visualize ranking differences with the following boxplot:

![model_rank_comparisons_with_all-MiniLM-L6-v2](https://github.com/htao00/NLP_similar_interests/assets/16727807/5bada166-470d-4274-b525-bb9c6bbabd07)

Lastly we can take a closer look at the top 5 people who's descriptions are the most similar to mine in all models:
<img width="1023" alt="Screenshot 2023-11-10 at 13 11 50" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/4e2b47a4-1aa2-4280-8353-f5433df2a186">
<img width="1019" alt="Screenshot 2023-11-10 at 13 13 35" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/2592b216-65e1-4741-8ddb-fc0a1e1283e8">
<img width="1050" alt="Screenshot 2023-11-10 at 13 14 12" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/4e92c784-b470-497c-b10a-fa93cdcb7716">
<img width="1062" alt="Screenshot 2023-11-10 at 13 14 47" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/78c514e8-1b54-4470-9860-22178831a37f">
<img width="1028" alt="Screenshot 2023-11-10 at 13 15 21" src="https://github.com/htao00/NLP_similar_interests/assets/16727807/1dcbeecf-0bc3-4060-bbcb-4e07c28ca3d9">


## References
1. Nilesh Barla, MLOps Blog: The Ultimate Guide to Word Embeddings, Neptune AI, 2023
   https://neptune.ai/blog/word-embeddings-guide
2. Diogo Ferreira, What Are Sentence Embeddings and why Are They Useful?, Medium, 2020
   https://engineering.talkdesk.com/what-are-sentence-embeddings-and-why-are-they-useful-53ed370b3f35
3. Sanjeev Arora et al., A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS, ICLR 2017
   https://openreview.net/pdf?id=SyK00v5xx
4. 
