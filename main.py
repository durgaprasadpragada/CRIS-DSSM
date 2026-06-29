"""
CRIS DSSM - Main Entry Point
Cryptocurrency Risk Insight System using Dynamic State Space Modeling

This module provides both CLI and programmatic interfaces to the CRIS DSSM system.
"""
import logging
import argparse
from typing import List
from pipeline import create_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_articles(articles: List[str]) -> dict:
    """
    Analyze news articles using CRIS DSSM

    Args:
        articles: List of news articles to analyze

    Returns:
        Dictionary with risk analysis results
    """
    pipeline = create_pipeline()
    return pipeline.run(articles)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='CRIS DSSM - Cryptocurrency Risk Insight System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start interactive dashboard
  python main.py --dashboard
  
  # Analyze articles from file
  python main.py --articles news.txt
  
  # Test with sample article
  python main.py --sample
        '''
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start interactive Dash dashboard'
    )

    parser.add_argument(
        '--articles',
        type=str,
        help='Path to file containing articles (one per line)'
    )

    parser.add_argument(
        '--sample',
        action='store_true',
        help='Run with sample articles'
    )

    args = parser.parse_args()

    # Start dashboard
    if args.dashboard:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

    # Analyze from file
    elif args.articles:
        logger.info(f"Reading articles from {args.articles}")
        try:
            with open(args.articles, 'r', encoding='utf-8') as f:
                articles = [line.strip() for line in f if line.strip()]

            if not articles:
                logger.error("No articles found in file")
                return

            logger.info(f"Analyzing {len(articles)} articles...")
            results = analyze_articles(articles)

            # Print results
            print("\n" + "="*60)
            print("CRIS DSSM Analysis Results")
            print("="*60)
            print(f"Coins analyzed: {results.get('coins_analyzed', 0)}")
            print(f"Articles processed: {results.get('articles_analyzed', 0)}")

            if results.get('highest_risk_coins'):
                print("\nHighest Risk Coins:")
                for i, coin in enumerate(results['highest_risk_coins'][:5], 1):
                    print(f"  {i}. {coin['coin']}: {coin['current_risk']:.2%} "
                          f"({coin.get('risk_category', 'Unknown')})")

            if results.get('model_metrics'):
                print("\nModel Metrics:")
                metrics = results['model_metrics']
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        print(f"  {key}: mean={val.get('mean', 'N/A'):.4f}, "
                              f"std={val.get('std', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"Error: {e}")

    # Sample analysis
    elif args.sample:
        sample_articles = [
            "Bitcoin ETF approval delayed by SEC regulatory concerns",
            "Ethereum upgrades face delays as developers work on scaling",
            "Major banks adopt Ripple technology for cross-border payments",
            "Crypto market rebounds as Fed signals rate pause",
        ]

        logger.info("Running sample analysis...")
        results = analyze_articles(sample_articles)

        print("\n" + "="*60)
        print("Sample CRIS DSSM Analysis Results")
        print("="*60)
        print(f"Coins analyzed: {results.get('coins_analyzed', 0)}")
        print(f"Articles processed: {results.get('articles_analyzed', 0)}")

        if results.get('highest_risk_coins'):
            print("\nHighest Risk Coins:")
            for i, coin in enumerate(results['highest_risk_coins'][:5], 1):
                print(f"  {i}. {coin['coin']}: {coin['current_risk']:.2%} "
                      f"({coin.get('risk_category', 'Unknown')})")

        if results.get('model_metrics'):
            print("\nModel Metrics:")
            metrics = results['model_metrics']
            for key, val in metrics.items():
                if isinstance(val, dict):
                    print(f"  {key}: mean={val.get('mean', 'N/A'):.4f}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
