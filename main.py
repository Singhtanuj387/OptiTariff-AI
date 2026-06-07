"""Master Orchestration Script - Agentic AI Dynamic Tariff Optimization.

Runs the complete pipeline:
  Phase 1: Data Preprocessing & Feature Engineering
  Phase 2: Exploratory Data Analysis
  Phase 3: Demand Prediction Agent
  Phase 4: Tariff Pricing Agent
  Phase 5: Monitoring & Learning Agent
  Phase 6: Evaluation & Reporting
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all():
    """Run the complete pipeline."""
    start_time = time.time()
    
    print('\n' + '='*70)
    print('  AGENTIC AI DYNAMIC TARIFF OPTIMIZATION FOR EV CHARGING NETWORKS')
    print('  Using Large-Scale Charging Session Data')
    print('='*70)
    
    # Phase 1: Preprocessing
    from src.data_preprocessing import run_preprocessing
    acn_sessions, acn_hourly, uev_hourly = run_preprocessing()
    
    # Phase 1B: Feature Engineering
    from src.feature_engineering import run_feature_engineering
    fe, acn_featured, uev_featured = run_feature_engineering(acn_sessions, acn_hourly, uev_hourly)
    
    # Phase 2: EDA
    from src.eda import run_eda
    run_eda(acn_sessions, acn_featured, uev_featured)
    
    # Phase 3: Demand Prediction Agent (on UrbanEV - primary dataset)
    from src.demand_prediction_agent import run_demand_prediction
    feature_cols = fe.get_ml_features('urbanev')
    demand_agent, predictions_df = run_demand_prediction(uev_featured, feature_cols, 'urbanev')
    
    # Also run on ACN
    if acn_featured is not None:
        acn_feature_cols = fe.get_ml_features('acn')
        run_demand_prediction(acn_featured, acn_feature_cols, 'acn')
    
    # Phase 4: Tariff Pricing Agent
    from src.tariff_pricing_agent import run_tariff_pricing
    tariff_agent, priced_df = run_tariff_pricing(predictions_df, 'urbanev')
    
    # Phase 5: Monitoring & Learning Agent
    from src.monitoring_agent import run_monitoring
    monitor_agent, episode_results = run_monitoring(priced_df, 'urbanev')
    
    # Phase 6: Evaluation
    from src.evaluation import run_evaluation
    run_evaluation('urbanev')
    
    elapsed = time.time() - start_time
    print(f'\n\n{"="*70}')
    print(f'  PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  Outputs saved to: outputs/')
    print(f'{"="*70}\n')


def main():
    parser = argparse.ArgumentParser(description='EV Charging Dynamic Tariff Optimization')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing only')
    parser.add_argument('--eda', action='store_true', help='Run EDA only')
    parser.add_argument('--demand-agent', action='store_true', help='Run demand prediction')
    parser.add_argument('--tariff-agent', action='store_true', help='Run tariff pricing')
    parser.add_argument('--monitor-agent', action='store_true', help='Run monitoring agent')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    
    args = parser.parse_args()
    
    if args.all or not any(vars(args).values()):
        run_all()
    else:
        if args.preprocess:
            from src.data_preprocessing import run_preprocessing
            from src.feature_engineering import run_feature_engineering
            acn_sessions, acn_hourly, uev_hourly = run_preprocessing()
            run_feature_engineering(acn_sessions, acn_hourly, uev_hourly)
        if args.eda:
            from src.eda import run_eda
            run_eda()
        if args.demand_agent:
            from src.demand_prediction_agent import run_demand_prediction
            run_demand_prediction()
        if args.tariff_agent:
            from src.tariff_pricing_agent import run_tariff_pricing
            run_tariff_pricing()
        if args.monitor_agent:
            from src.monitoring_agent import run_monitoring
            run_monitoring()
        if args.evaluate:
            from src.evaluation import run_evaluation
            run_evaluation()


if __name__ == '__main__':
    main()
