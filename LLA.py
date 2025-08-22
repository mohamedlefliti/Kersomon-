# krezomon
# Krezomon: A Hybrid Rule-Based and ML Framework for Transparent Startup Evaluation
# With Fraud & Governance Risk Detection
# All outputs in English

import math
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from xgboost import XGBClassifier
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# === 1. User Growth Score (0-1)
def calculate_user_growth(current_users: int, previous_users: int) -> float:
    if previous_users == 0:
        return 0.0
    growth_rate = (current_users - previous_users) / previous_users
    return min(1.0, 1 - math.exp(-3 * growth_rate))

# === 2. Team Quality Score
def evaluate_team(founders: List[Dict]) -> float:
    if not founders:
        return 0.0
    total_score = 0.0
    for founder in founders:
        score = 0.0
        score += 0.2 if founder.get('previous_exit', False) else 0.0
        score += 0.3 * min(founder.get('domain_experience_years', 0) / 5, 1.0)
        score += 0.3 if founder.get('technical_background', False) else 0.0
        score += 0.2 if founder.get('academic_research', False) else 0.0
        total_score += score
    return total_score / len(founders)

# === 3. Investor Quality
def investor_quality(investors: List[str]) -> float:
    tier1 = {'sequoia', 'andreessen horowitz', 'accel', 'benchmark', 'y combinator', 'softbank', 'kpcb', 'greylock'}
    tier2 = {'founders fund', 'union square', 'first round', 'true ventures'}

    quality_score = 0.0
    for inv in investors:
        inv_lower = inv.lower()
        if inv_lower in tier1:
            quality_score += 1.0
        elif inv_lower in tier2:
            quality_score += 0.7
        elif any(kw in inv_lower for kw in ['capital', 'ventures', 'partners']):
            quality_score += 0.3
        else:
            quality_score += 0.5 if 'angel' in inv_lower else 0.4
    return min(quality_score / 3, 1.0)

# === 4. Market Potential
def market_potential(market_growth_rate: float, competitor_count: int) -> float:
    growth_component = 1 - math.exp(-3 * market_growth_rate)
    competition_component = math.exp(-0.2 * competitor_count)
    network_bonus = 1.5 if market_growth_rate > 0.5 else 1.0
    return min((0.6 * growth_component + 0.4 * competition_component) * network_bonus, 1.0)

# === 5. Innovation Strength
def innovation_strength(rnd_budget: float, patents: int, tech_advantage: str, market_impact: str) -> float:
    tech_multiplier = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
    impact_multiplier = {'transformative': 1.0, 'disruptive': 0.7, 'incremental': 0.3}

    rnd_component = min(rnd_budget / 1.0, 1.0)
    patent_component = 1 - math.exp(-0.2 * patents)

    return (
        0.25 * rnd_component +
        0.15 * patent_component +
        0.30 * tech_multiplier.get(tech_advantage.lower(), 0) +
        0.30 * impact_multiplier.get(market_impact.lower(), 0)
    )

# === 6. Financial Health
def financial_health(burn_rate: float, runway: int, revenue_growth: float, gross_margin: float) -> float:
    burn_component = math.exp(-0.8 * burn_rate)
    runway_component = min(runway / 24.0, 1.0)
    growth_component = 1 - math.exp(-3 * revenue_growth)
    margin_component = min(gross_margin / 0.7, 1.0) if gross_margin > 0 else 0.5
    return (0.3 * burn_component + 0.3 * runway_component + 0.2 * growth_component + 0.2 * margin_component)

# === 7. Media Impact
def media_impact(press: List[str], social_eng: float, industry_recognition: bool) -> float:
    quality_outlets = {'techcrunch', 'wired', 'the verge', 'mit technology review', 'wsj', 'ny times', 'bloomberg'}
    coverage_score = sum(1 for outlet in press if any(q in outlet.lower() for q in quality_outlets))
    coverage_component = min(coverage_score / 2.0, 1.0)
    social_component = min(social_eng / 500000, 1.0)
    awards_component = 0.3 if industry_recognition else 0.0
    return min(0.5 * coverage_component + 0.2 * social_component + awards_component, 1.0)

# === 8. Fraud & Governance Risk Layer ===

def transparency_score(
    audited_financials: bool = False,
    open_technical_whitepaper: bool = False,
    third_party_validations: int = 0,
    data_access_for_researchers: bool = False,
    clear_roadmap: bool = False
) -> float:
    """Transparency score: higher = more trustworthy"""
    return (
        0.3 * (1.0 if audited_financials else 0.0) +
        0.2 * (1.0 if open_technical_whitepaper else 0.0) +
        0.2 * min(third_party_validations / 3, 1.0) +
        0.15 * (1.0 if data_access_for_researchers else 0.0) +
        0.15 * (1.0 if clear_roadmap else 0.0)
    )

def governance_risk(
    founder_has_conflict_of_interest: bool = False,
    related_party_transactions: bool = False,
    lack_of_board_independence: bool = False,
    opaque_corporate_structure: bool = False,
    founder_controls_funds: bool = False
) -> float:
    """Governance risk score: higher = more risk"""
    risk = 0.0
    if founder_has_conflict_of_interest: risk += 0.25
    if related_party_transactions: risk += 0.25
    if lack_of_board_independence: risk += 0.2
    if opaque_corporate_structure: risk += 0.2
    if founder_controls_funds: risk += 0.1
    return min(risk, 1.0)

def hype_to_product_ratio(media_coverage_count: int, product_maturity: str) -> float:
    """Hype-to-product ratio: higher = more risk"""
    hype_score = min(media_coverage_count / 10, 1.0)
    maturity_map = {'prototype': 0.2, 'beta': 0.5, 'mature': 1.0}
    product_score = maturity_map.get(product_maturity.lower(), 0.2)
    return hype_score / (product_score + 0.05)

def founder_centric_risk(team_size: int, media_mentions_founder: int, technical_founder_ratio: float) -> float:
    """Founder-centric risk: over-reliance on founder persona"""
    risk = 0.0
    if media_mentions_founder > 50:
        risk += 0.3
    if technical_founder_ratio < 0.5:
        risk += 0.3
    if team_size < 10 and media_mentions_founder > 50:
        risk += 0.4
    return min(risk, 1.0)

# === 9. Rule-Based Decision (Krezomon v2) ===
def investment_decision_v2(startup: Dict) -> Dict:
    scores = {
        'user_growth': calculate_user_growth(startup['current_users'], startup['previous_users']),
        'team_quality': evaluate_team(startup['founders']),
        'investor_quality': investor_quality(startup['investors']),
        'market_potential': market_potential(startup['market_growth_rate'], startup['competitor_count']),
        'innovation': innovation_strength(startup['rnd_budget'], startup['patents'],
                                       startup['tech_advantage'], startup.get('market_impact', 'disruptive')),
        'financials': financial_health(startup['burn_rate'], startup['runway_months'],
                                     startup.get('revenue_growth', 0.0), startup.get('gross_margin', 0.0)),
        'media': media_impact(startup['press_coverage'], startup.get('social_engagement', 0),
                            startup.get('industry_awards', False))
    }

    # Dynamic weights based on revenue stage
    if startup.get('revenue_growth', 0) > 0.5 or startup.get('revenue_current', 0) > 0:
        weights = {
            'user_growth': 0.15, 'team_quality': 0.15, 'investor_quality': 0.15,
            'market_potential': 0.20, 'innovation': 0.15, 'financials': 0.15, 'media': 0.05
        }
    else:
        weights = {
            'user_growth': 0.25, 'team_quality': 0.20, 'investor_quality': 0.15,
            'market_potential': 0.20, 'innovation': 0.15, 'financials': 0.05, 'media': 0.05
        }

    base_score = sum(scores[factor] * weights[factor] for factor in scores)

    # Bonus for breakthrough innovation
    if scores['innovation'] > 0.8 and scores['market_potential'] > 0.7:
        base_score = min(base_score * 1.2, 1.0)

    # === Fraud & Governance Layer ===
    transparency = transparency_score(
        audited_financials=startup.get('audited_financials', False),
        open_technical_whitepaper=startup.get('open_technical_whitepaper', False),
        third_party_validations=startup.get('third_party_validations', 0),
        data_access_for_researchers=startup.get('data_access_for_researchers', False),
        clear_roadmap=startup.get('clear_roadmap', False)
    )

    gov_risk = governance_risk(
        founder_has_conflict_of_interest=startup.get('founder_has_conflict_of_interest', False),
        related_party_transactions=startup.get('related_party_transactions', False),
        lack_of_board_independence=startup.get('lack_of_board_independence', False),
        opaque_corporate_structure=startup.get('opaque_corporate_structure', False),
        founder_controls_funds=startup.get('founder_controls_funds', False)
    )

    hype_ratio = hype_to_product_ratio(
        media_coverage_count=len(startup['press_coverage']),
        product_maturity=startup.get('product_maturity', 'prototype')
    )

    founder_risk = founder_centric_risk(
        team_size=len(startup['founders']),
        media_mentions_founder=startup.get('media_mentions_founder', 0),
        technical_founder_ratio=startup.get('technical_founder_ratio', 0.3)
    )

    # Apply governance penalties
    if gov_risk > 0.6 or transparency < 0.3:
        final_score = max(base_score * 0.5, 0.3)
        confidence = "High (Governance Risk Detected)"
    elif gov_risk > 0.4 or transparency < 0.5:
        final_score = base_score * 0.7
        confidence = "Moderate (Transparency Concern)"
    else:
        final_score = base_score
        confidence = "High"

    if final_score >= 0.75:
        decision = "STRONG BUY"
    elif final_score >= 0.60:
        decision = "BUY"
    elif final_score >= 0.45:
        decision = "WATCH"
    else:
        decision = "PASS"

    return {
        'decision': decision,
        'confidence': confidence,
        'weighted_score': round(final_score, 2),
        'component_scores': {k: round(v, 2) for k, v in scores.items()},
        'fraud_risk_indicators': {
            'transparency_score': round(transparency, 2),
            'governance_risk': round(gov_risk, 2),
            'hype_to_product_ratio': round(hype_ratio, 2),
            'founder_centric_risk': round(founder_risk, 2)
        }
    }

# === 10. Ethical Gate: Hard Filter for Fraud & Governance ===
def ethical_gate(fraud_indicators: Dict) -> Dict:
    """
    Strict ethical filter: blocks investment if critical red flags exist.
    """
    transparency = fraud_indicators['transparency_score']
    gov_risk = fraud_indicators['governance_risk']
    hype_ratio = fraud_indicators['hype_to_product_ratio']
    founder_risk = fraud_indicators['founder_centric_risk']

    red_flags = []
    blocked = False

    if transparency < 0.4:
        red_flags.append("LOW_TRANSPARENCY: audited financials, whitepaper, or validations missing")
        blocked = True
    if gov_risk > 0.6:
        red_flags.append("HIGH_GOVERNANCE_RISK: conflicts, related-party transactions, or fund control")
        blocked = True
    if hype_ratio > 1.0:
        red_flags.append("EXCESSIVE_HYPE: media coverage far exceeds product maturity")
        blocked = True
    if founder_risk > 0.7:
        red_flags.append("FOUNDER_CENTRIC_RISK: over-reliance on non-technical founder with excessive media focus")
        blocked = True

    return {
        'passed': not blocked,
        'red_flags': red_flags,
        'verdict': "BLOCK" if blocked else "CLEAR",
        'details': {
            'transparency_score': transparency,
            'governance_risk': gov_risk,
            'hype_to_product_ratio': hype_ratio,
            'founder_centric_risk': founder_risk
        }
    }

# === 11. Generate Synthetic Dataset (200+ startups) ===
def generate_synthetic_data(n=200):
    np.random.seed(42)
    data = []
    for i in range(n):
        success = np.random.rand() > 0.3  # 70% success rate

        current_users = int(np.exp(np.random.normal(8, 2)))
        previous_users = int(current_users * np.random.uniform(0.5, 0.9))
        revenue_growth = np.random.normal(0.3, 0.2) if success else np.random.normal(0.1, 0.3)
        revenue_growth = max(revenue_growth, 0)

        transparency = np.random.normal(0.7, 0.2) if success else np.random.normal(0.3, 0.2)
        transparency = np.clip(transparency, 0, 1)
        gov_risk = np.random.normal(0.2, 0.1) if success else np.random.normal(0.6, 0.2)
        gov_risk = np.clip(gov_risk, 0, 1)

        hype = np.random.normal(0.5, 0.2) if success else np.random.normal(1.2, 0.3)
        product = np.random.normal(0.7, 0.1) if success else np.random.normal(0.4, 0.2)
        hype_to_product = np.clip(hype / (product + 0.1), 0, 3)

        data.append([
            current_users, previous_users, revenue_growth,
            np.random.randint(5, 50),  # team_size
            np.random.randint(1, 5),   # founder_count
            np.random.normal(8, 3),    # avg_domain_exp
            np.random.normal(0.7, 0.2),  # technical_founder_ratio
            int(np.random.rand() > 0.7),
            np.random.randint(0, 3),
            np.random.choice([2, 3]),  # investor_tier
            np.random.normal(0.3, 0.15),  # market_growth_rate
            np.random.randint(3, 12),  # competitor_count
            np.random.normal(1.0, 0.5),  # rnd_budget_millions
            np.random.randint(0, 50),  # patents
            np.random.choice([1, 2, 3]),  # tech_advantage
            np.random.choice([1, 2, 3]),  # market_impact
            np.random.normal(0.8, 0.3),  # burn_rate_millions
            np.random.randint(12, 36),  # runway_months
            np.random.normal(0.5, 0.2),  # gross_margin
            np.random.randint(1, 10),  # press_count
            np.random.normal(300000, 200000),  # social_engagement
            int(np.random.rand() > 0.3),
            int(transparency > 0.5),
            int(transparency > 0.6),
            int(np.random.normal(3, 1)),  # third_party_validations
            int(transparency > 0.7),
            int(transparency > 0.5),
            int(gov_risk > 0.5),
            int(gov_risk > 0.4),
            int(gov_risk < 0.5),
            int(gov_risk > 0.6),
            int(gov_risk > 0.7),
            np.random.choice([1, 2, 3]),  # product_maturity
            int(np.random.normal(50, 30)),  # media_mentions_founder
            hype_to_product,
            int(success)
        ])
    return data

# === 12. Train Krezomon v3 (ML Model) ===
def train_krezomon_v3():
    columns = [
        'current_users', 'previous_users', 'revenue_growth', 'team_size', 'founder_count',
        'avg_domain_exp', 'technical_founder_ratio', 'founder_has_exit', 'academic_research_founders',
        'investor_tier', 'market_growth_rate', 'competitor_count', 'rnd_budget_millions', 'patents',
        'tech_advantage', 'market_impact', 'burn_rate_millions', 'runway_months', 'gross_margin',
        'press_count', 'social_engagement', 'industry_awards', 'audited_financials', 'open_whitepaper',
        'third_party_validations', 'data_access_researchers', 'clear_roadmap', 'founder_conflict',
        'related_party_transactions', 'board_independence', 'opaque_structure', 'founder_controls_funds',
        'product_maturity', 'media_mentions_founder', 'hype_to_product_ratio', 'survival_3y'
    ]

    data = generate_synthetic_data(200)
    df = pd.DataFrame(data, columns=columns)
    X = df.drop('survival_3y', axis=1)
    y = df['survival_3y']

    model = XGBClassifier(objective='binary:logistic', max_depth=4, n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "krezomon_v3_model.pkl")
    print("✅ Krezomon v3 model trained and saved as 'krezomon_v3_model.pkl'")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("\n🔍 Top Features (SHAP Importance):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))

    return model, explainer

# === 13. Hybrid Decision (v2 + v3) with Ethical Gate ===
def hybrid_investment_decision(startup: Dict, ml_model, ml_explainer):
    v2_result = investment_decision_v2(startup)
    v2_score = v2_result['weighted_score']

    # Apply strict ethical gate first
    ethics = ethical_gate(v2_result['fraud_risk_indicators'])

    if not ethics['passed']:
        return {
            'decision': "BLOCKED_FOR_ETHICS",
            'hybrid_score': 0.0,
            'ml_prediction': 0.0,
            'explanation': "Investment blocked due to critical ethical and governance risks.",
            'ethical_verdict': ethics,
            'fraud_indicators': v2_result['fraud_risk_indicators'],
            'component_scores': v2_result['component_scores']
        }

    # Map startup to ML features
    features = pd.DataFrame([{
        'current_users': startup['current_users'],
        'previous_users': startup['previous_users'],
        'revenue_growth': startup.get('revenue_growth', 0),
        'team_size': len(startup['founders']),
        'founder_count': len(startup['founders']),
        'avg_domain_exp': np.mean([f.get('domain_experience_years', 0) for f in startup['founders']]),
        'technical_founder_ratio': sum(1 for f in startup['founders'] if f.get('technical_background')) / len(startup['founders']),
        'founder_has_exit': int(any(f.get('previous_exit') for f in startup['founders'])),
        'academic_research_founders': sum(1 for f in startup['founders'] if f.get('academic_research')),
        'investor_tier': 3 if 'sequoia' in str(startup['investors']).lower() else 2,
        'market_growth_rate': startup['market_growth_rate'],
        'competitor_count': startup['competitor_count'],
        'rnd_budget_millions': startup['rnd_budget'],
        'patents': startup['patents'],
        'tech_advantage': {'high': 3, 'medium': 2, 'low': 1}.get(startup['tech_advantage'].lower(), 2),
        'market_impact': {'transformative': 3, 'disruptive': 2, 'incremental': 1}.get(startup.get('market_impact', 'disruptive').lower(), 2),
        'burn_rate_millions': startup['burn_rate'],
        'runway_months': startup['runway_months'],
        'gross_margin': startup.get('gross_margin', 0),
        'press_count': len(startup['press_coverage']),
        'social_engagement': startup.get('social_engagement', 0),
        'industry_awards': int(startup.get('industry_awards', False)),
        'audited_financials': int(startup.get('audited_financials', False)),
        'open_whitepaper': int(startup.get('open_technical_whitepaper', False)),
        'third_party_validations': startup.get('third_party_validations', 0),
        'data_access_researchers': int(startup.get('data_access_for_researchers', False)),
        'clear_roadmap': int(startup.get('clear_roadmap', False)),
        'founder_conflict': int(startup.get('founder_has_conflict_of_interest', False)),
        'related_party_transactions': int(startup.get('related_party_transactions', False)),
        'board_independence': int(not startup.get('lack_of_board_independence', False)),
        'opaque_structure': int(startup.get('opaque_corporate_structure', False)),
        'founder_controls_funds': int(startup.get('founder_controls_funds', False)),
        'product_maturity': {'prototype': 1, 'beta': 2, 'mature': 3}.get(startup.get('product_maturity', 'prototype'), 1),
        'media_mentions_founder': startup.get('media_mentions_founder', 0),
        'hype_to_product_ratio': hype_to_product_ratio(
            media_coverage_count=len(startup['press_coverage']),
            product_maturity=startup.get('product_maturity', 'prototype')
        )
    }])

    # Predict with ML model
    ml_proba = ml_model.predict_proba(features)[0][1]
    v3_score = ml_proba

    # Hybrid score (60% rule-based, 40% ML)
    hybrid_score = 0.6 * v2_score + 0.4 * v3_score

    # SHAP explanation
    shap_values = ml_explainer.shap_values(features)
    top_indices = np.argsort(np.abs(shap_values[0]))[-5:]
    top_features = features.columns[top_indices].tolist()

    return {
        'decision': v2_result['decision'],
        'hybrid_score': round(hybrid_score, 2),
        'ml_prediction': round(v3_score, 2),
        'explanation': f"The decision is supported by: {', '.join(top_features)}. High integrity standards applied.",
        'ethical_verdict': ethics,
        'fraud_indicators': v2_result['fraud_risk_indicators'],
        'component_scores': v2_result['component_scores']
    }

# === 14. Run Example ===
if __name__ == "__main__":
    print("🚀 Training Krezomon v3 Model...")
    ml_model, ml_explainer = train_krezomon_v3()

    # === 1. NeuroSync Robotics ===
    neurosync_2025 = {
        'name': 'NeuroSync Robotics',
        'current_users': 150, 'previous_users': 50,
        'founders': [
            {'name': 'Dr. Layla', 'previous_exit': False, 'technical_background': True, 'domain_experience_years': 8, 'academic_research': True},
            {'name': 'Dr. Karim', 'technical_background': True, 'domain_experience_years': 7, 'academic_research': True},
            {'name': 'Dr. Sami', 'technical_background': True, 'domain_experience_years': 6, 'academic_research': True},
            {'name': 'Ahmed', 'technical_background': True, 'domain_experience_years': 9, 'academic_research': True}
        ],
        'investors': ['Sequoia', 'Y Combinator'],
        'market_growth_rate': 0.45, 'competitor_count': 8,
        'rnd_budget': 0.8, 'patents': 12, 'tech_advantage': 'high', 'market_impact': 'transformative',
        'burn_rate': 0.6, 'runway_months': 18, 'revenue_growth': 0.0, 'gross_margin': 0.0,
        'press_coverage': ['TechCrunch', 'MIT Technology Review', 'Wired'],
        'social_engagement': 400000, 'industry_awards': True,
        'audited_financials': True, 'open_technical_whitepaper': True,
        'third_party_validations': 5, 'data_access_for_researchers': True, 'clear_roadmap': True,
        'founder_has_conflict_of_interest': False, 'related_party_transactions': False,
        'lack_of_board_independence': False, 'opaque_corporate_structure': False, 'founder_controls_funds': False,
        'product_maturity': 'beta', 'media_mentions_founder': 30
    }

    print("\n\n=== Evaluating NeuroSync Robotics ===")
    result = hybrid_investment_decision(neurosync_2025, ml_model, ml_explainer)
    print(f"Decision: {result['decision']}")
    if result['decision'] == "BLOCKED_FOR_ETHICS":
        print("🔴 Status: ETHICAL BLOCK")
        print("Red Flags:")
        for flag in result['ethical_verdict']['red_flags']:
            print(f"  - {flag}")
    else:
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"ML Prediction: {result['ml_prediction']}")
        print(f"Explanation: {result['explanation']}")
        print("Fraud Risk Indicators:")
        for k, v in result['fraud_indicators'].items():
            print(f"  - {k}: {v}")

    # === 2. Suspicious Startup: VitaLume Health ===
    suspicious_startup = {
        'name': 'VitaLume Health',
        'current_users': 200,
        'previous_users': 150,
        'founders': [
            {'name': 'Alex Rayan', 'previous_exit': False, 'technical_background': False, 'domain_experience_years': 2, 'academic_research': False},
        ],
        'investors': ['Prominent Angel Investor', 'Family Office'],
        'market_growth_rate': 0.4,
        'competitor_count': 6,
        'rnd_budget': 0.1,
        'patents': 1,
        'tech_advantage': 'high',
        'market_impact': 'transformative',
        'burn_rate': 1.2,
        'runway_months': 8,
        'revenue_growth': 0.0,
        'gross_margin': 0.0,
        'press_coverage': ['Forbes', 'CNBC', 'Bloomberg', 'Wired', 'TechCrunch', 'The Verge', 'WSJ', 'Time', 'BI', 'Fast Co'],
        'social_engagement': 5000000,
        'industry_awards': True,
        'audited_financials': False,
        'open_technical_whitepaper': False,
        'third_party_validations': 0,
        'data_access_for_researchers': False,
        'clear_roadmap': False,
        'founder_has_conflict_of_interest': True,
        'related_party_transactions': True,
        'lack_of_board_independence': True,
        'opaque_corporate_structure': True,
        'founder_controls_funds': True,
        'product_maturity': 'prototype',
        'media_mentions_founder': 150,
        'technical_founder_ratio': 0.0
    }

    print("\n\n=== Evaluating Suspicious Startup: VitaLume Health ===")
    result_suspicious = hybrid_investment_decision(suspicious_startup, ml_model, ml_explainer)
    print(f"Decision: {result_suspicious['decision']}")
    if result_suspicious['decision'] == "BLOCKED_FOR_ETHICS":
        print("🔴 Status: ETHICAL BLOCK")
        print("Red Flags:")
        for flag in result_suspicious['ethical_verdict']['red_flags']:
            print(f"  - {flag}")
    else:
        print(f"Hybrid Score: {result_suspicious['hybrid_score']}")
        print(f"ML Prediction: {result_suspicious['ml_prediction']}")
        print(f"Explanation: {result_suspicious['explanation']}")
        print("Fraud Risk Indicators:")
        for k, v in result_suspicious['fraud_indicators'].items():
            print(f"  - {k}: {v}")


# === 15. Evaluate 10 Diverse Startups ===
print("\n\n" + "="*60)
print("       🌍 EVALUATING 10 DIVERSE STARTUPS")
print("="*60)

startups = [
    {
        'name': 'GreenField AgriTech',
        'current_users': 25000,
        'previous_users': 18000,
        'founders': [
            {'name': 'Dr. Amina', 'technical_background': True, 'domain_experience_years': 12, 'academic_research': True},
            {'name': 'Karim', 'technical_background': True, 'domain_experience_years': 8, 'previous_exit': True}
        ],
        'investors': ['AgriVentures', 'World Food Fund'],
        'market_growth_rate': 0.18,
        'competitor_count': 7,
        'rnd_budget': 0.3,
        'patents': 4,
        'tech_advantage': 'medium',
        'market_impact': 'incremental',
        'burn_rate': 0.2,
        'runway_months': 30,
        'revenue_growth': 0.4,
        'gross_margin': 0.5,
        'press_coverage': ['Forbes Agri', 'TechNode', 'The Farm Weekly'],
        'social_engagement': 120000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 3,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'mature',
        'media_mentions_founder': 12,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'MediScan AI',
        'current_users': 5000,
        'previous_users': 2000,
        'founders': [
            {'name': 'Dr. Zhang', 'technical_background': True, 'domain_experience_years': 15, 'academic_research': True},
            {'name': 'Lena', 'technical_background': True, 'domain_experience_years': 10}
        ],
        'investors': ['Sequoia', 'HealthX Fund'],
        'market_growth_rate': 0.35,
        'competitor_count': 9,
        'rnd_budget': 1.5,
        'patents': 18,
        'tech_advantage': 'high',
        'market_impact': 'transformative',
        'burn_rate': 2.0,
        'runway_months': 14,
        'revenue_growth': 0.0,
        'gross_margin': 0.0,
        'press_coverage': ['Nature Medicine', 'MIT Tech Review', 'WSJ Health'],
        'social_engagement': 800000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 5,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'beta',
        'media_mentions_founder': 45,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'Solaris Dynamics',
        'current_users': 300,
        'previous_users': 100,
        'founders': [
            {'name': 'Carlos', 'technical_background': True, 'domain_experience_years': 20, 'previous_exit': True},
            {'name': 'Elena', 'technical_background': True, 'domain_experience_years': 18}
        ],
        'investors': ['Energy Future Fund', 'EU Green Tech'],
        'market_growth_rate': 0.25,
        'competitor_count': 5,
        'rnd_budget': 3.0,
        'patents': 25,
        'tech_advantage': 'high',
        'market_impact': 'disruptive',
        'burn_rate': 1.8,
        'runway_months': 22,
        'revenue_growth': 0.3,
        'gross_margin': 0.6,
        'press_coverage': ['Renewable Energy Today', 'CleanTech News'],
        'social_engagement': 90000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 6,
        'data_access_for_researchers': False,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'beta',
        'media_mentions_founder': 20,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'EduNova Platform',
        'current_users': 120000,
        'previous_users': 90000,
        'founders': [
            {'name': 'Sarah', 'technical_background': True, 'domain_experience_years': 7},
            {'name': 'David', 'technical_background': True, 'domain_experience_years': 6}
        ],
        'investors': ['LearnFund', 'UNESCO EdTech'],
        'market_growth_rate': 0.22,
        'competitor_count': 15,
        'rnd_budget': 0.5,
        'patents': 2,
        'tech_advantage': 'medium',
        'market_impact': 'incremental',
        'burn_rate': 0.4,
        'runway_months': 36,
        'revenue_growth': 0.5,
        'gross_margin': 0.7,
        'press_coverage': ['EdSurge', 'The PIE News'],
        'social_engagement': 300000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 4,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'mature',
        'media_mentions_founder': 8,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'QuantumLeap AI',
        'current_users': 1000,
        'previous_users': 800,
        'founders': [
            {'name': 'Victor', 'technical_background': False, 'domain_experience_years': 4, 'previous_exit': False},
        ],
        'investors': ['Future Capital', 'Visionary Syndicate'],
        'market_growth_rate': 0.5,
        'competitor_count': 12,
        'rnd_budget': 0.2,
        'patents': 0,
        'tech_advantage': 'high',
        'market_impact': 'transformative',
        'burn_rate': 4.0,
        'runway_months': 6,
        'revenue_growth': 0.0,
        'gross_margin': 0.0,
        'press_coverage': ['Forbes', 'Bloomberg', 'CNBC', 'Wired', 'TechCrunch'] * 2,
        'social_engagement': 10000000,
        'industry_awards': True,
        'audited_financials': False,
        'open_technical_whitepaper': False,
        'third_party_validations': 0,
        'data_access_for_researchers': False,
        'clear_roadmap': False,
        'founder_has_conflict_of_interest': True,
        'related_party_transactions': True,
        'lack_of_board_independence': True,
        'opaque_corporate_structure': True,
        'founder_controls_funds': True,
        'product_maturity': 'prototype',
        'media_mentions_founder': 200,
        'technical_founder_ratio': 0.0
    },
    {
        'name': 'BioSafe Diagnostics',
        'current_users': 8000,
        'previous_users': 3000,
        'founders': [
            {'name': 'Dr. Liu', 'technical_background': True, 'domain_experience_years': 14, 'academic_research': True},
            {'name': 'Dr. Chen', 'technical_background': True, 'domain_experience_years': 12, 'academic_research': True}
        ],
        'investors': ['MedInvest', 'Global Health Fund'],
        'market_growth_rate': 0.3,
        'competitor_count': 8,
        'rnd_budget': 2.0,
        'patents': 15,
        'tech_advantage': 'high',
        'market_impact': 'disruptive',
        'burn_rate': 1.5,
        'runway_months': 18,
        'revenue_growth': 0.1,
        'gross_margin': 0.4,
        'press_coverage': ['Nature Biotech', 'STAT News'],
        'social_engagement': 200000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 7,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'beta',
        'media_mentions_founder': 35,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'CryptoNova Chain',
        'current_users': 50000,
        'previous_users': 40000,
        'founders': [
            {'name': 'Max', 'technical_background': True, 'domain_experience_years': 3, 'previous_exit': False},
        ],
        'investors': ['CryptoFund Alpha', 'DeFi Syndicate'],
        'market_growth_rate': 0.8,
        'competitor_count': 20,
        'rnd_budget': 0.3,
        'patents': 1,
        'tech_advantage': 'medium',
        'market_impact': 'incremental',
        'burn_rate': 5.0,
        'runway_months': 4,
        'revenue_growth': 0.0,
        'gross_margin': 0.0,
        'press_coverage': ['CoinDesk', 'Cointelegraph', 'Decrypt'] * 3,
        'social_engagement': 15000000,
        'industry_awards': True,
        'audited_financials': False,
        'open_technical_whitepaper': False,
        'third_party_validations': 0,
        'data_access_for_researchers': False,
        'clear_roadmap': False,
        'founder_has_conflict_of_interest': True,
        'related_party_transactions': True,
        'lack_of_board_independence': True,
        'opaque_corporate_structure': True,
        'founder_controls_funds': True,
        'product_maturity': 'prototype',
        'media_mentions_founder': 180,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'CleanAir Sensors',
        'current_users': 15000,
        'previous_users': 12000,
        'founders': [
            {'name': 'Prof. Nour', 'technical_background': True, 'domain_experience_years': 16, 'academic_research': True},
            {'name': 'Ali', 'technical_background': True, 'domain_experience_years': 9}
        ],
        'investors': ['Green Earth Fund', 'Smart Cities Inc'],
        'market_growth_rate': 0.2,
        'competitor_count': 6,
        'rnd_budget': 0.6,
        'patents': 8,
        'tech_advantage': 'medium',
        'market_impact': 'incremental',
        'burn_rate': 0.3,
        'runway_months': 32,
        'revenue_growth': 0.35,
        'gross_margin': 0.55,
        'press_coverage': ['Environmental Tech Review', 'IoT Today'],
        'social_engagement': 75000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 4,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'mature',
        'media_mentions_founder': 15,
        'technical_founder_ratio': 1.0
    },
    {
        'name': 'NeuroLink Mind',
        'current_users': 200,
        'previous_users': 100,
        'founders': [
            {'name': 'Dr. Orion', 'technical_background': False, 'domain_experience_years': 1, 'previous_exit': False},
        ],
        'investors': ['Futuristic Capital', 'MetaFund'],
        'market_growth_rate': 0.6,
        'competitor_count': 5,
        'rnd_budget': 0.1,
        'patents': 0,
        'tech_advantage': 'high',
        'market_impact': 'transformative',
        'burn_rate': 3.0,
        'runway_months': 5,
        'revenue_growth': 0.0,
        'gross_margin': 0.0,
        'press_coverage': ['The Future', 'Next Big Thing', 'Innovation Weekly'] * 4,
        'social_engagement': 20000000,
        'industry_awards': True,
        'audited_financials': False,
        'open_technical_whitepaper': False,
        'third_party_validations': 0,
        'data_access_for_researchers': False,
        'clear_roadmap': False,
        'founder_has_conflict_of_interest': True,
        'related_party_transactions': True,
        'lack_of_board_independence': True,
        'opaque_corporate_structure': True,
        'founder_controls_funds': True,
        'product_maturity': 'prototype',
        'media_mentions_founder': 250,
        'technical_founder_ratio': 0.0
    },
    {
        'name': 'FarmConnect Africa',
        'current_users': 80000,
        'previous_users': 50000,
        'founders': [
            {'name': 'Kwame', 'technical_background': True, 'domain_experience_years': 10},
            {'name': 'Amina', 'technical_background': True, 'domain_experience_years': 8},
            {'name': 'David', 'technical_background': False, 'domain_experience_years': 12, 'academic_research': False}
        ],
        'investors': ['AfriFund', 'UNDP Impact'],
        'market_growth_rate': 0.15,
        'competitor_count': 4,
        'rnd_budget': 0.2,
        'patents': 1,
        'tech_advantage': 'low',
        'market_impact': 'incremental',
        'burn_rate': 0.15,
        'runway_months': 40,
        'revenue_growth': 0.6,
        'gross_margin': 0.7,
        'press_coverage': ['African Business', 'Rural Tech News'],
        'social_engagement': 100000,
        'industry_awards': True,
        'audited_financials': True,
        'open_technical_whitepaper': True,
        'third_party_validations': 5,
        'data_access_for_researchers': True,
        'clear_roadmap': True,
        'founder_has_conflict_of_interest': False,
        'related_party_transactions': False,
        'lack_of_board_independence': False,
        'opaque_corporate_structure': False,
        'founder_controls_funds': False,
        'product_maturity': 'mature',
        'media_mentions_founder': 10,
        'technical_founder_ratio': 0.67
    }
]

# Evaluate all 10
for i, startup in enumerate(startups, 1):
    print(f"\n--- [{i}] Evaluating: {startup['name']} ---")
    result = hybrid_investment_decision(startup, ml_model, ml_explainer)
    print(f"Decision: {result['decision']}")
    if result['decision'] == "BLOCKED_FOR_ETHICS":
        print("🔴 Status: ETHICAL BLOCK")
        for flag in result['ethical_verdict']['red_flags']:
            print(f"  - {flag}")
    else:
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"Fraud Indicators: Hype={result['fraud_indicators']['hype_to_product_ratio']}, GovRisk={result['fraud_indicators']['governance_risk']}")
