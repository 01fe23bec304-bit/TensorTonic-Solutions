def precision_recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    
    relevant_count = 0
    for item in top_k:
        if item in relevant:
            relevant_count += 1
    
    precision = relevant_count / k if k > 0 else 0
    recall = relevant_count / len(relevant) if len(relevant) > 0 else 0
    
    return [float(precision), float(recall)]