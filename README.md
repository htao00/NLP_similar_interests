# NLP_similar_interests
NLP assignment on finding MCDA classmates with similar interests

## What are Embeddings?
Word and sentence embeddings are a clever way of turning words and sentences into numerical representations that computers can understand. Word and sentence embeddings make it possible for computers to work with and analyze text, making tasks like language translation, sentiment analysis, and information retrieval much easier and more accurate. A common way to numerically represent words and sentences is to transform them into vectors. Vectors are essentially a bunch of numbers arranged in a specific order, and they represent different aspects of the word or sentence.

For word embeddings, imagine each word as a unique vector. These vectors capture the word's meaning in a multi-dimensional space. Each dimension might represent something like word frequency, context, or similarity to other words. So, when we take a word and transform it into a vector, we're assigning it a specific location in this multi-dimensional space, sort of like plotting points on a map.
![image](https://github.com/htao00/NLP_similar_interests/assets/16727807/f8b2c191-501b-4e17-be8d-96457403a32a)
We expect related words or words with similar meaning to be used in similar contexts and appear with similar number of frequencies. Therefore, once tranformed into vectors we should see the similarities reflected in vector space. As shown in the plot above, related words are clustered close with each other.

Sentence embeddings are like puzzles made up of word vectors. When we form a sentence, we combine the individual word vectors in a specific way to create a new vector that represents the entire sentence. This combined vector captures the overall meaning and context of the sentence by taking into account the relationships between the words.
![image](https://github.com/htao00/NLP_similar_interests/assets/16727807/68e6f07b-6f1b-40ce-b389-798a30fd03bc)
The closer the vectors are, the more similar the words or sentences are in meaning. So, word and sentence embeddings are like a bridge that helps computers make sense of our words and language.
