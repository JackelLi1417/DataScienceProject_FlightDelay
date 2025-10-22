#!/usr/bin/env python3
"""
完整的航班取消预测系统 - 包含特征重要性分析

新增功能：
- 特征重要性排名
- 天气特征 vs 其他特征对比
- 可视化特征重要性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings

# 机器学习相关
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# 处理不平衡数据
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings('ignore')


# ==================== 配置日志 ====================
def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('FlightPredictor')


logger = setup_logger()


# ==================== 特征定义 ====================
FEATURE_NAMES = [
    # 天气特征 (10个)
    'HourlyDryBulbTemperature_x', 'HourlyDryBulbTemperature_y',
    'HourlyPrecipitation_x', 'HourlyPrecipitation_y',
    'HourlyStationPressure_x', 'HourlyStationPressure_y',
    'HourlyVisibility_x', 'HourlyVisibility_y',
    'HourlyWindSpeed_x', 'HourlyWindSpeed_y',
    # 机场ID (2个)
    'STATION_x', 'STATION_y',
    # 延误特征 (8个)
    'arrival_delay', 'delay_carrier', 'delay_late_aircarft_arrival',
    'delay_national_aviation_system', 'delay_security', 'delay_weather',
    'departure_delay', 'scheduled_elapsed_time',
    # 时间特征 (3个)
    'month', 'weekday', 'year'
]

WEATHER_FEATURES = [
    'HourlyDryBulbTemperature_x', 'HourlyDryBulbTemperature_y',
    'HourlyPrecipitation_x', 'HourlyPrecipitation_y',
    'HourlyStationPressure_x', 'HourlyStationPressure_y',
    'HourlyVisibility_x', 'HourlyVisibility_y',
    'HourlyWindSpeed_x', 'HourlyWindSpeed_y'
]

# 特征中文名映射
FEATURE_NAME_CN = {
    'HourlyDryBulbTemperature_x': '起飞地温度',
    'HourlyDryBulbTemperature_y': '目的地温度',
    'HourlyPrecipitation_x': '起飞地降水量',
    'HourlyPrecipitation_y': '目的地降水量',
    'HourlyStationPressure_x': '起飞地气压',
    'HourlyStationPressure_y': '目的地气压',
    'HourlyVisibility_x': '起飞地能见度',
    'HourlyVisibility_y': '目的地能见度',
    'HourlyWindSpeed_x': '起飞地风速',
    'HourlyWindSpeed_y': '目的地风速',
    'STATION_x': '起飞机场',
    'STATION_y': '目的地机场',
    'arrival_delay': '到达延误',
    'delay_carrier': '航司延误',
    'delay_late_aircarft_arrival': '前序航班延误',
    'delay_national_aviation_system': '系统延误',
    'delay_security': '安检延误',
    'delay_weather': '天气延误',
    'departure_delay': '出发延误',
    'scheduled_elapsed_time': '计划飞行时间',
    'month': '月份',
    'weekday': '星期',
    'year': '年份'
}


# ==================== 数据加载 ====================
def load_flight_data(data_dir='archive', sample_fraction=0.1):
    """Load flight data"""
    logger.info("="*60)
    logger.info("Loading data...")
    logger.info("="*60)
    
    data_path = Path(data_dir)
    months = ['05', '06', '07', '08', '09', '10', '11', '12']
    dataframes = []
    
    for month in months:
        file_path = data_path / f'{month}-2019.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            dataframes.append(df)
            logger.info(f"✓ Loaded {month}-2019.csv: {len(df):,} records")
        else:
            logger.warning(f"✗ File not found: {file_path}")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"\nTotal records: {len(combined_df):,}")
    
    sampled_df = combined_df.sample(frac=sample_fraction, random_state=42)
    logger.info(f"After sampling: {len(sampled_df):,} records ({sample_fraction*100}%)")
    
    return sampled_df


# ==================== 特征工程 ====================
def prepare_features(df, binary_classification=True):
    """Prepare features and target variable"""
    logger.info("\n" + "="*60)
    logger.info("Feature Engineering...")
    logger.info("="*60)
    
    target_col = 'cancelled_code'
    
    logger.info(f"Total features: {len(FEATURE_NAMES)}")
    logger.info(f"  • Weather features: 10 (43.5%)")
    logger.info(f"  • Delay features: 8 (34.8%)")
    logger.info(f"  • Temporal features: 3 (13.0%)")
    logger.info(f"  • Other features: 2 (8.7%)")
    
    # Select features and clean data
    df_clean = df[FEATURE_NAMES + [target_col]].dropna()
    logger.info(f"\nAfter cleaning: {len(df_clean):,} records")
    
    # Prepare X and y
    X = df_clean[FEATURE_NAMES].values
    y_raw = df_clean[target_col].values
    
    # Handle target variable
    if binary_classification:
        logger.info("\n📊 Using **Binary Classification** (Cancelled vs Not Cancelled)")
        y = np.where(y_raw == 'N', 0, 1)
        classes = np.array(['Not Cancelled', 'Cancelled'])
    else:
        logger.info("\n📊 Using **Multiclass Classification** (4 classes)")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        classes = label_encoder.classes_
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"\nClass Distribution:")
    for cls, count in zip(unique, counts):
        pct = count / len(y) * 100
        logger.info(f"  {classes[cls]:15s}: {count:6,} ({pct:5.2f}%)")
    
    if len(counts) > 1:
        imbalance_ratio = max(counts) / min(counts)
        logger.info(f"\n⚠️  Imbalance Ratio: 1:{imbalance_ratio:.1f}")
    
    return X, y, classes


# ==================== 数据预处理 ====================
def preprocess_data(X, y, test_size=0.2, apply_smote=True):
    """Data preprocessing: split, normalize, oversample"""
    logger.info("\n" + "="*60)
    logger.info("Data Preprocessing...")
    logger.info("="*60)
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"\nData Split:")
    logger.info(f"  Training set: {len(X_train):,} samples")
    logger.info(f"  Test set: {len(X_test):,} samples")
    
    # 2. Normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"\n✓ Feature normalization complete (MinMaxScaler)")
    
    # 3. SMOTE oversampling
    if apply_smote:
        logger.info(f"\nApplying SMOTE oversampling...")
        
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"  Before oversampling:")
        for cls, count in zip(unique, counts):
            logger.info(f"    Class {cls}: {count:,}")
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        logger.info(f"  After oversampling:")
        for cls, count in zip(unique, counts):
            logger.info(f"    Class {cls}: {count:,}")
        
        logger.info(f"\n  Sample count: {len(X_train_scaled):,} → {len(X_train_resampled):,}")
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==================== 模型训练 ====================
def train_models(X_train, y_train):
    """Train multiple models"""
    logger.info("\n" + "="*60)
    logger.info("Training Models...")
    logger.info("="*60)
    
    models = {}
    
    # Model 1: Random Forest with class weights
    logger.info("\n1️⃣  Random Forest (class_weight='balanced')")
    rf_balanced = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        max_features=0.3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_balanced.fit(X_train, y_train)
    models['RF_Balanced'] = rf_balanced
    logger.info("   ✓ Training complete")
    
    # Model 2: Balanced Random Forest
    logger.info("\n2️⃣  Balanced Random Forest (BalancedRandomForest)")
    brf = BalancedRandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    brf.fit(X_train, y_train)
    models['BalancedRF'] = brf
    logger.info("   ✓ Training complete")
    
    # Model 3: Regular Random Forest
    logger.info("\n3️⃣  Regular Random Forest (baseline)")
    rf_regular = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        max_features=0.3,
        random_state=42,
        n_jobs=-1
    )
    rf_regular.fit(X_train, y_train)
    models['RF_Regular'] = rf_regular
    logger.info("   ✓ Training complete")
    
    logger.info(f"\n✅ Trained {len(models)} models")
    
    return models


# ==================== 特征重要性分析 ====================
def analyze_feature_importance(models):
    """Analyze feature importance"""
    logger.info("\n" + "="*60)
    logger.info("📊 Feature Importance Analysis")
    logger.info("="*60)
    
    all_importances = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            all_importances[model_name] = importances
    
    # Use best model's importance
    best_model_name = 'RF_Balanced'
    importances = all_importances[best_model_name]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Feature_CN': [FEATURE_NAME_CN[f] for f in FEATURE_NAMES],
        'Importance': importances,
        'Is_Weather': [f in WEATHER_FEATURES for f in FEATURE_NAMES]
    })
    
    # Sort
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Print Top 15
    logger.info(f"\n🏆 Top 15 Most Important Features (Model: {best_model_name}):")
    logger.info("-" * 70)
    logger.info(f"{'Rank':<5} {'Feature Name':<35} {'Importance':>10} {'Type':>10}")
    logger.info("-" * 70)
    
    for idx, row in importance_df.head(15).iterrows():
        feature_type = "Weather" if row['Is_Weather'] else "Other"
        logger.info(f"{importance_df.index.get_loc(idx)+1:<5} {row['Feature']:<35} {row['Importance']:>10.4f} {feature_type:>10}")
    
    # Weather vs Other features
    weather_importance = importance_df[importance_df['Is_Weather']]['Importance'].sum()
    other_importance = importance_df[~importance_df['Is_Weather']]['Importance'].sum()
    
    logger.info("\n" + "="*70)
    logger.info("📊 Weather Features vs Other Features:")
    logger.info(f"  Weather Features Total Importance: {weather_importance:.4f} ({weather_importance*100:.2f}%)")
    logger.info(f"  Other Features Total Importance: {other_importance:.4f} ({other_importance*100:.2f}%)")
    logger.info("="*70)
    
    # Most important weather features
    weather_df = importance_df[importance_df['Is_Weather']].head(5)
    logger.info(f"\n🌦️  Top 5 Most Important Weather Features:")
    logger.info("-" * 70)
    for idx, row in weather_df.iterrows():
        logger.info(f"  {row['Feature']:<35} {row['Importance']:>10.4f}")
    
    return importance_df


def plot_feature_importance(importance_df, save_dir='outputs'):
    """Plot feature importance charts (English only)"""
    logger.info("\nGenerating feature importance visualizations...")
    
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Chart 1: Top 20 Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Left: Top 20 all features
    ax = axes[0]
    top_20 = importance_df.head(20)
    colors = ['#FF6B6B' if is_weather else '#4ECDC4' 
              for is_weather in top_20['Is_Weather']]
    
    bars = ax.barh(range(len(top_20)), top_20['Importance'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Feature'])  # Use English feature names
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Weather Features'),
        Patch(facecolor='#4ECDC4', label='Other Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Right: Weather vs Other Features
    ax = axes[1]
    weather_importance = importance_df[importance_df['Is_Weather']]['Importance'].sum()
    other_importance = importance_df[~importance_df['Is_Weather']]['Importance'].sum()
    
    categories = ['Weather Features\n(10 features)', 'Other Features\n(13 features)']
    values = [weather_importance, other_importance]
    colors_pie = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=categories, 
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {save_path}")
    plt.close()
    
    # Chart 2: Detailed Weather Feature Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    weather_df = importance_df[importance_df['Is_Weather']].sort_values('Importance', ascending=True)
    
    bars = ax.barh(range(len(weather_df)), weather_df['Importance'], color='#FF6B6B', edgecolor='darkred')
    ax.set_yticks(range(len(weather_df)))
    ax.set_yticklabels(weather_df['Feature'])  # Use English feature names
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Weather Features Importance Detailed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, weather_df['Importance'])):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{importance:.4f}',
               ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    save_path = output_dir / 'weather_feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {save_path}")
    plt.close()


# ==================== 模型评估 ====================
def evaluate_model(model_name, model, X_test, y_test, classes):
    """Evaluate a single model"""
    logger.info("\n" + "-"*60)
    logger.info(f"Evaluating: {model_name}")
    logger.info("-"*60)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"\n📊 Overall Metrics:")
    logger.info(f"   Accuracy:          {accuracy:.4f}")
    logger.info(f"   Balanced Accuracy: {balanced_acc:.4f}  ⭐")
    logger.info(f"   Weighted F1-Score: {f1_weighted:.4f}")
    
    logger.info(f"\n📋 Per-Class Metrics:")
    for i in range(len(classes)):
        logger.info(f"\n   【{classes[i]}】")
        logger.info(f"      Precision: {precision[i]:.4f}")
        logger.info(f"      Recall:    {recall[i]:.4f}  {'⭐' if i == 1 else ''}")
        logger.info(f"      F1-Score:  {f1[i]:.4f}")
        logger.info(f"      Support:   {support[i]}")
    
    logger.info(f"\n🎯 Confusion Matrix:")
    logger.info("\n" + str(conf_matrix))
    
    if len(classes) == 2:
        minority_recall = recall[1]
        logger.info(f"\n💡 Model can identify {minority_recall*100:.1f}% of cancelled flights!")
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    return results


# ==================== 可视化 ====================
def plot_results(all_results, classes, save_dir='outputs'):
    """Plot result comparison charts (English only)"""
    logger.info("\n" + "="*60)
    logger.info("Generating model performance visualizations...")
    logger.info("="*60)
    
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_names = list(all_results.keys())
    
    # Chart 1: Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    ax = axes[0, 0]
    accuracies = [all_results[name]['accuracy'] for name in model_names]
    bars = ax.bar(model_names, accuracies, color='skyblue', edgecolor='navy')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Balanced Accuracy
    ax = axes[0, 1]
    balanced_accs = [all_results[name]['balanced_accuracy'] for name in model_names]
    bars = ax.bar(model_names, balanced_accs, color='lightgreen', edgecolor='darkgreen')
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Balanced Accuracy Comparison (Most Important)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1-Score
    ax = axes[1, 0]
    f1_scores = [all_results[name]['f1_weighted'] for name in model_names]
    bars = ax.bar(model_names, f1_scores, color='lightsalmon', edgecolor='darkred')
    ax.set_ylabel('Weighted F1-Score', fontsize=12)
    ax.set_title('Weighted F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Minority Class Recall
    ax = axes[1, 1]
    if len(classes) == 2:
        recalls = [all_results[name]['recall'][1] for name in model_names]
        ax.bar(model_names, recalls, color='gold', edgecolor='darkorange')
        ax.set_ylabel('Recall (Cancelled Flights)', fontsize=12)
        ax.set_title('Recall for Cancelled Flights (Critical Metric)', fontsize=14, fontweight='bold')
    else:
        recalls = [np.mean(all_results[name]['recall'][:-1]) for name in model_names]
        ax.bar(model_names, recalls, color='gold', edgecolor='darkorange')
        ax.set_ylabel('Avg Recall (Minority Classes)', fontsize=12)
        ax.set_title('Average Recall for Minority Classes', fontsize=14, fontweight='bold')
    
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, recall) in enumerate(zip(ax.patches, recalls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = output_dir / 'model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {save_path}")
    plt.close()
    
    # Chart 2: Confusion Matrices
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        cm = all_results[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 Flight Cancellation Prediction System - Full Version")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ SMOTE oversampling")
    print("  ✓ Class weight balancing")
    print("  ✓ Binary classification")
    print("  ✓ Correct evaluation metrics")
    print("  ✓ Feature importance analysis ⭐")
    print("\n" + "="*60 + "\n")
    
    try:
        # 1. Load data
        df = load_flight_data(data_dir='archive', sample_fraction=0.1)
        
        # 2. Feature engineering
        X, y, classes = prepare_features(df, binary_classification=True)
        
        # 3. Data preprocessing
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X, y, test_size=0.2, apply_smote=True
        )
        
        # 4. Train models
        models = train_models(X_train, y_train)
        
        # 5. Feature importance analysis ⭐
        importance_df = analyze_feature_importance(models)
        plot_feature_importance(importance_df)
        
        # 6. Evaluate all models
        logger.info("\n" + "="*60)
        logger.info("Model Evaluation")
        logger.info("="*60)
        
        all_results = {}
        for model_name, model in models.items():
            results = evaluate_model(model_name, model, X_test, y_test, classes)
            all_results[model_name] = results
        
        # 7. Visualization
        plot_results(all_results, classes)
        
        # 8. Summary
        logger.info("\n" + "="*60)
        logger.info("📋 Final Summary")
        logger.info("="*60)
        
        best_model = max(all_results.keys(), 
                        key=lambda x: all_results[x]['balanced_accuracy'])
        best_results = all_results[best_model]
        
        logger.info(f"\n🏆 Best Model: {best_model}")
        logger.info(f"   Balanced Accuracy: {best_results['balanced_accuracy']:.4f}")
        logger.info(f"   Weighted F1-Score: {best_results['f1_weighted']:.4f}")
        
        if len(classes) == 2:
            minority_recall = best_results['recall'][1]
            logger.info(f"   Recall (Cancelled Flights): {minority_recall:.4f}")
            logger.info(f"\n💡 This model can identify {minority_recall*100:.1f}% of cancelled flights!")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Pipeline execution complete!")
        logger.info("📊 Check the 'outputs' folder for charts:")
        logger.info("   • model_comparison.png - Model performance comparison")
        logger.info("   • confusion_matrices.png - Confusion matrices")
        logger.info("   • feature_importance.png - Feature importance ⭐")
        logger.info("   • weather_feature_importance.png - Weather features detail ⭐")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
