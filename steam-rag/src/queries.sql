-- Count reviews per topic/sentiment with filters
SELECT t.topic,
       t.sentiment,
       COUNT(*) AS review_count
FROM review_topics t
JOIN reviews r ON r.review_id = t.review_id
WHERE r.app_id = :app_id
  AND r.lang = 'english'
  AND r.created_at BETWEEN :start AND :end
  AND r.helpful_count >= :min_helpful
GROUP BY t.topic, t.sentiment
ORDER BY t.topic, t.sentiment;

-- Top negative reviews by helpfulness for a topic
SELECT r.review_id,
       r.helpful_count,
       r.review_text
FROM review_topics t
JOIN reviews r ON r.review_id = t.review_id
WHERE r.app_id = :app_id
  AND t.topic = :topic
  AND t.sentiment IN ('negative', 'very_negative')
  AND r.created_at BETWEEN :start AND :end
ORDER BY r.helpful_count DESC
LIMIT 20;

-- Monthly trend by topic and sentiment
SELECT date_trunc('month', r.created_at) AS month,
       t.topic,
       t.sentiment,
       COUNT(*) AS review_count
FROM review_topics t
JOIN reviews r ON r.review_id = t.review_id
WHERE r.app_id = :app_id
  AND r.helpful_count >= :min_helpful
GROUP BY month, t.topic, t.sentiment
ORDER BY month, t.topic, t.sentiment;
