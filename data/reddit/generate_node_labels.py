import pandas as pd
from sklearn.cluster import KMeans

def generate_node_labels(node_file, out_file, num_classes=10):
    print(f"Loading node features from: {node_file}")
    df = pd.read_csv(node_file)

    if 'id' not in df.columns:
        df.columns = ['id'] + [f'f{i}' for i in range(1, df.shape[1])]

    node_ids = df['id']
    feats = df.drop(columns=['id']).values

    print(f"Clustering into {num_classes} classes...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    labels = kmeans.fit_predict(feats)

    label_df = pd.DataFrame({
        'id': node_ids,
        'label': labels,
        'time': 0  # static label
    })

    label_df.to_csv(out_file, index=False)
    print(f"Saved label file to: {out_file}")


if __name__ == '__main__':
    node_file = './data/reddit/web-redditEmbeddings-subreddits.csv'
    out_file = './data/reddit/node_labels.csv'
    generate_node_labels(node_file, out_file, num_classes=7)
