import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_virality_trends(memes_data):
    if not memes_data:
        print("No meme data provided for plotting.")
        return

    os.makedirs("data", exist_ok=True)

    try:
        df = pd.DataFrame(memes_data)
        df['date'] = pd.to_datetime(df['metadata'].apply(lambda x: x['date']))
        df['virality'] = (df['metadata'].apply(lambda x: x['likes'] + x['retweets'])) / ((pd.Timestamp.now() - df['date']).dt.days + 1)
        df['trend_keyword'] = df['text'].str.lower().str.contains('pandas|rag|sql', na=False).map({True: 'Hot Topics', False: 'General'})

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='trend_keyword', y='virality', hue='trend_keyword', palette='deep')
        plt.title("Meme Virality in Data Analytics (Last 7 Days)")
        plt.ylabel("Avg Virality Score")
        plt.savefig("data/virality_plot.png")
        plt.close()
    except Exception as e:
        print(f"Error generating virality plot: {str(e)}")