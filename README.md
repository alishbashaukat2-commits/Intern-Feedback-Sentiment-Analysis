## Final Conclusion

In this task, intern feedback was analyzed using sentiment classification to understand overall intern satisfaction and identify areas for improvement. The dataset contained 1,500 feedback records, consisting of 520 positive, 500 negative, and 480 neutral reviews.

Two approaches were evaluated. Logistic Regression with TF-IDF achieved an accuracy of **99%** and a macro F1-score of **0.99**. A pretrained RoBERTa Transformer, used without fine-tuning, achieved an accuracy of **77%** and a macro F1-score of **0.76**. The Transformer showed more difficulty distinguishing neutral feedback, while Logistic Regression performed better on this particular dataset.

The Transformer was not trained from scratch because the dataset was relatively small and the available laptop environment was CPU-based. Training a Transformer from scratch would require substantially more data and computational resources. Therefore, a pretrained Transformer was used as a practical baseline.

Analysis of negative and neutral feedback highlighted several areas that could improve intern satisfaction. These included better mentor communication and guidance, more meaningful and less repetitive tasks, greater opportunities for learning new tools and skills, career-development guidance, more engaging training, and better management of workplace pressure.

Overall, the sentiment analysis provided useful information about intern experiences and highlighted specific areas that organizations can consider when improving their internship programs.
