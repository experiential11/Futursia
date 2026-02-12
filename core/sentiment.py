"""Sentiment analysis for news headlines."""

import re
from datetime import datetime


class SimpleSentimentAnalyzer:
    """Simple rule-based sentiment analyzer for news headlines."""
    
    # Positive and negative keywords
    POSITIVE_WORDS = {
        'bullish', 'surge', 'soar', 'jump', 'rally', 'gains', 'growth', 'profit',
        'beat', 'upgrade', 'outperform', 'strong', 'recovery', 'support', 'bull',
        'rally', 'upside', 'positive', 'boom', 'surge', 'skyrocket', 'win',
        'approval', 'deal', 'partnership', 'breakthrough', 'success', 'record',
        'milestone', 'expansion', 'innovation', 'leadership'
    }
    
    NEGATIVE_WORDS = {
        'bearish', 'crash', 'plunge', 'falls', 'losses', 'decline', 'drop', 'loss',
        'miss', 'downgrade', 'underperform', 'weak', 'slump', 'resistance', 'bear',
        'selloff', 'downside', 'negative', 'crash', 'collapse', 'bankruptcy',
        'downfall', 'warning', 'threat', 'risk', 'scandal', 'investigation',
        'lawsuit', 'recall', 'layoff', 'shutdown', 'restructure'
    }
    
    def analyze_headline(self, headline):
        """Analyze sentiment of a headline.
        
        Args:
            headline: Headline text
        
        Returns:
            dict: {'sentiment': -1/0/1, 'score': float, 'explanation': str}
        """
        headline_lower = headline.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in headline_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in headline_lower)
        
        # Boost if words are at beginning (more important)
        first_words = ' '.join(headline_lower.split()[:3])
        positive_count += 2 * sum(1 for word in self.POSITIVE_WORDS if word in first_words)
        negative_count += 2 * sum(1 for word in self.NEGATIVE_WORDS if word in first_words)
        
        # Determine sentiment
        net_score = positive_count - negative_count
        
        if net_score > 0:
            sentiment = 1
            explanation = f"Positive (+{positive_count} words)"
        elif net_score < 0:
            sentiment = -1
            explanation = f"Negative (+{negative_count} words)"
        else:
            sentiment = 0
            explanation = "Neutral"
        
        # Normalize score to -1 to 1
        total = positive_count + negative_count
        if total > 0:
            score = (positive_count - negative_count) / total
        else:
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': float(score),
            'explanation': explanation,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def aggregate_sentiment(self, headlines, lookback_minutes=60):
        """Aggregate sentiment across multiple headlines.
        
        Args:
            headlines: List of headline dicts with 'text' and 'timestamp' keys
            lookback_minutes: Only consider recent headlines
        
        Returns:
            dict: Aggregated sentiment metrics
        """
        if not headlines:
            return {
                'overall_sentiment': 0,
                'score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'headline_count': 0
            }
        
        now = datetime.utcnow()
        cutoff = now.timestamp() - (lookback_minutes * 60)
        
        # Filter recent headlines
        recent = []
        for h in headlines:
            try:
                h_time = h.get('timestamp', now)
                if isinstance(h_time, str):
                    h_time = datetime.fromisoformat(h_time.replace('Z', '+00:00'))
                if isinstance(h_time, datetime):
                    h_time = h_time.timestamp()
                
                if h_time >= cutoff:
                    recent.append(h)
            except Exception:
                recent.append(h)
        
        # Analyze each headline
        sentiments = []
        scores = []
        
        for h in recent:
            text = h.get('text', h.get('title', ''))
            if text:
                analysis = self.analyze_headline(text)
                sentiments.append(analysis['sentiment'])
                scores.append(analysis['score'])
        
        if not sentiments:
            return {
                'overall_sentiment': 0,
                'score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'headline_count': 0
            }
        
        # Aggregate
        positive_count = sum(1 for s in sentiments if s > 0)
        negative_count = sum(1 for s in sentiments if s < 0)
        neutral_count = sum(1 for s in sentiments if s == 0)
        
        avg_score = sum(scores) / len(scores)
        
        # Overall sentiment
        if positive_count > negative_count:
            overall_sentiment = 1
        elif negative_count > positive_count:
            overall_sentiment = -1
        else:
            overall_sentiment = 0
        
        return {
            'overall_sentiment': overall_sentiment,
            'score': float(avg_score),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'headline_count': len(sentiments)
        }
