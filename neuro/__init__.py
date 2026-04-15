"""
neuro — нейросетевой модуль матчинга товаров.

Архитектура: Siamese Transformer Encoder + FAISS.
Обучение: Triplet Loss с hard negative mining.
Инференс: < 5ms на 1M записей (CPU).
"""
