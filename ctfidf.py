from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class CTFIDFProcessor:
    def __init__(self, max_features=5000, stop_words='english'):
        """
        Initialize the CTFIDFProcessor with optional parameters.
        :param max_features: Number of most frequent terms to consider
        :param stop_words: Stop words to remove (default: English)
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
        self.tfidf_matrix = None
        self.feature_names = None
        self.class_tfidf = None
    
    def fit_transform(self, documents, labels):
        """
        Fit the c-TF-IDF model on the given documents and transform them.
        :param documents: List of text documents
        :param labels: Corresponding class labels for each document
        :return: c-TF-IDF matrix (DataFrame)
        """
        df = pd.DataFrame({'document': documents, 'class': labels})
        class_docs = df.groupby('class')['document'].apply(lambda x: ' '.join(x))
        self.tfidf_matrix = self.vectorizer.fit_transform(class_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.class_tfidf = pd.DataFrame(self.tfidf_matrix.toarray(), index=class_docs.index, columns=self.feature_names)
        return self.class_tfidf
    
    def transform(self, new_documents):
        """
        Transform new documents using the fitted c-TF-IDF model.
        :param new_documents: List of new text documents
        :return: Transformed c-TF-IDF matrix
        """
        return self.vectorizer.transform(new_documents)
    
    def get_feature_names(self):
        """
        Get the feature names (words) used in the c-TF-IDF representation.
        :return: List of feature names
        """
        return self.feature_names
    
    def get_ctfidf_matrix_as_dataframe(self):
        """
        Convert the c-TF-IDF matrix into a pandas DataFrame.
        :return: DataFrame containing c-TF-IDF values
        """
        if self.class_tfidf is not None:
            return self.class_tfidf
        else:
            raise ValueError("c-TF-IDF matrix is not yet computed. Run fit_transform first.")
    
    def extract_keywords(self, top_n=10):
        """
        Extract top-n important keywords for each class based on c-TF-IDF scores.
        :param top_n: Number of top keywords to extract per class
        :return: Dictionary with class labels as keys and lists of top-n keywords as values
        """
        if self.class_tfidf is None:
            raise ValueError("c-TF-IDF matrix is not yet computed. Run fit_transform first.")
        
        keywords = {}
        for class_label, row in self.class_tfidf.iterrows():
            top_keywords = row.nlargest(top_n).index.tolist()
            keywords[class_label] = top_keywords
        return keywords

# Example usage
if __name__ == "__main__":
    docs = [
        "Machine learning is amazing.",
        "Deep learning is a subset of machine learning.",
        "Natural language processing is a field of AI.",
        "AI and deep learning are advancing rapidly.",
        "Machine learning is used in various applications."
    ]
    labels = ["ML", "ML", "NLP", "AI", "ML"]
    
    ctfidf_processor = CTFIDFProcessor()
    class_tfidf_matrix = ctfidf_processor.fit_transform(docs, labels)
    print("Feature Names:", ctfidf_processor.get_feature_names())
    print("c-TF-IDF Matrix:")
    print(ctfidf_processor.get_ctfidf_matrix_as_dataframe())
    
    # Extract important keywords
    keywords = ctfidf_processor.extract_keywords(top_n=5)
    print("Important Keywords:")
    print(keywords)
