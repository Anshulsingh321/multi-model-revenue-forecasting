import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import load_data
from feature_engineering import create_features
from models import train_models, FEATURES
from evaluation import evaluate_models
from utils import inject_css, show_recommendation

# -------------------------
# Setup
# -------------------------
st.set_page_config(page_title="Revenue Dashboard", layout="wide")
inject_css()

st.title("📊 Revenue Forecasting Dashboard")
st.markdown("### Clean insights. Better decisions.")

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if not uploaded_file:
    st.info("📂 Upload a dataset to unlock analysis tabs and insights.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "📊 Model Performance",
    "📈 Model Visualization",
    "🔮 Forecast",
    "🧠 Feature Importance"
])

if uploaded_file:
    df = load_data(uploaded_file)

    required_cols = {"date", "revenue", "cash", "total_liabilities", "trend"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # -------------------------
    # Feature Engineering
    # -------------------------
    df_feat = create_features(df)

    # -------------------------
    # Train Models
    # -------------------------
    results, ridge, sarima_fit, holt = train_models(df_feat)

    # -------------------------
    # Evaluation
    # -------------------------
    metrics_df, best_model = evaluate_models(results)

    # -------------------------
    # Feature Importance
    # -------------------------
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": ridge.coef_,
    })
    importance_df["Importance"] = importance_df["Coefficient"].abs()
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("📅 Start Date", df['date'].min().strftime("%Y-%m-%d"))
        col2.metric("📅 End Date", df['date'].max().strftime("%Y-%m-%d"))
        col3.metric("📊 Records", len(df))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Engineered Features")
        st.markdown("""
        - Revenue Lag → seasonality  
        - YoY Growth → trend  
        - Trend → external signal  
        - Quarter → seasonal pattern  
        - Cash & Liabilities → financial health  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TAB 2: MODEL PERFORMANCE
    # =========================
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Model Performance")

        st.dataframe(
            metrics_df.style
            .highlight_min(subset=["MAPE"], color="#D1FAE5")
            .format({"RMSE": "{:.2f}", "MAPE": "{:.2%}"})
        )

        # -------------------------
        # RMSE Chart
        # -------------------------
        fig_rmse = go.Figure()

        best_model_name = best_model["Model"]

        colors_rmse = ["#16A34A" if m == best_model_name else "#1f77b4" for m in metrics_df["Model"]]

        fig_rmse.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["RMSE"],
            name="RMSE",
            marker_color=colors_rmse,
            text=["Best" if m == best_model_name else "" for m in metrics_df["Model"]],
            textposition="outside"
        ))

        fig_rmse.update_layout(
            title="RMSE Comparison",
            xaxis_title="Model",
            yaxis_title="RMSE",
            height=300
        )

        st.plotly_chart(fig_rmse, use_container_width=True)

        # -------------------------
        # MAPE Chart
        # -------------------------
        fig_mape = go.Figure()

        colors_mape = ["#16A34A" if m == best_model_name else "#60A5FA" for m in metrics_df["Model"]]

        fig_mape.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["MAPE"],
            name="MAPE",
            marker_color=colors_mape,
            text=[f"{val:.2%}" if m != best_model_name else f"Best ({val:.2%})" for m, val in zip(metrics_df["Model"], metrics_df["MAPE"])],
            textposition="outside"
        ))

        fig_mape.update_layout(
            title="MAPE Comparison",
            xaxis_title="Model",
            yaxis_title="MAPE (%)",
            height=300
        )

        st.plotly_chart(fig_mape, use_container_width=True)

        # -------------------------
        # Performance Insight
        # -------------------------
        best_row = metrics_df.loc[metrics_df['Model'] == best_model_name].iloc[0]
        rmse_val = best_row['RMSE']
        mape_val = best_row['MAPE']
        insight_text = (
            f"{best_model_name} performs best with lowest error metrics — "
            f"RMSE: {rmse_val:.2f}, MAPE: {mape_val:.2%}. "
            f"This indicates more accurate and reliable revenue predictions compared to other models."
        )
        st.info(f"🧠 Insight: {insight_text}")

        show_recommendation(best_model["Model"])
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TAB 3: MODEL VISUALIZATION
    # =========================
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Model Visualization")

        # Initialize toggle state
        if "model_toggles" not in st.session_state:
            st.session_state.model_toggles = {name: True for name in results.keys()}

        st.markdown("**Select models to display:**")
        cols = st.columns(len(results))
        for i, name in enumerate(results.keys()):
            with cols[i]:
                st.session_state.model_toggles[name] = st.toggle(name, value=st.session_state.model_toggles.get(name, True))

        # Prepare base data
        sample_key = list(results.keys())[0]
        y_true = results[sample_key][0]
        dates = df_feat.loc[y_true.index, 'date']

        fig = go.Figure()

        # Color palette for models
        colors = {
            "Actual": "black",
            "Ridge": "#1f77b4",
            "SARIMA": "#ff7f0e",
            "Holt-Winters": "#2ca02c",
            "Hybrid": "#d62728"
        }

        best_model_name = best_model["Model"]

        # Actual line (thicker + bold)
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_true.values,
            mode='lines',
            name='Actual',
            line=dict(width=4, color=colors.get("Actual", "black"))
        ))

        # Add selected models
        for name, (y_true_i, y_pred) in results.items():
            if st.session_state.model_toggles.get(name, False):

                line_style = dict(
                    dash='solid' if name == best_model_name else 'dash',
                    width=3 if name == best_model_name else 2,
                    color=colors.get(name, None)
                )

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=y_pred,
                    mode='lines',
                    name=f"{name} {'(Best)' if name == best_model_name else ''}",
                    line=line_style
                ))

        # Highlight best model annotation
        fig.add_annotation(
            text=f"Best Model: {best_model_name}",
            xref="paper", yref="paper",
            x=0.01, y=0.95,
            showarrow=False,
            font=dict(size=12, color="green")
        )

        fig.update_layout(
            title="Model Comparison (Interactive)",
            xaxis_title="Date",
            yaxis_title="Revenue",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TAB 4: FORECAST
    # =========================
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Future Forecast (Next 4 Quarters)")

        last_date = df_feat['date'].iloc[-1]

        future_dates = pd.date_range(
            start=last_date + pd.offsets.QuarterEnd(),
            periods=4,
            freq='QE'
        )

        best_model_name = best_model["Model"]

        if best_model_name == "Holt-Winters":
            best_forecast = holt.forecast(4)
        elif best_model_name == "SARIMA":
            best_forecast = sarima_fit.get_forecast(steps=4).predicted_mean
        else:
            best_forecast = holt.forecast(4)

        sarima_forecast = sarima_fit.get_forecast(steps=4)
        conf_int = sarima_forecast.conf_int()

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": best_forecast.values,
            "Lower": conf_int.iloc[:, 0].values,
            "Upper": conf_int.iloc[:, 1].values
        })

        # Create clean quarter labels for x-axis
        future_df["Quarter"] = future_df["Date"].dt.to_period("Q").astype(str)

        # -------------------------
        # Forecast Chart
        # -------------------------
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=future_df["Quarter"],
            y=future_df["Forecast"],
            mode='lines+markers',
            name='Forecast',
            line=dict(dash='dash')
        ))

        fig_forecast.add_trace(go.Scatter(
            x=future_df["Quarter"].tolist() + future_df["Quarter"][::-1].tolist(),
            y=future_df["Upper"].tolist() + future_df["Lower"][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval'
        ))

        fig_forecast.update_layout(
            title="Forecast (Next 4 Quarters)",
            xaxis_title="Quarter",
            yaxis_title="Revenue",
            height=350,
            hovermode="x unified"
        )


        st.plotly_chart(fig_forecast, use_container_width=True)

        # -------------------------
        # Forecast Summary Metrics
        # -------------------------
        total_forecast = future_df["Forecast"].sum()
        avg_forecast = future_df["Forecast"].mean()

        col1, col2 = st.columns(2)
        col1.metric("📊 Total Forecast", f"{total_forecast:,.0f}")
        col2.metric("📈 Avg Quarterly Revenue", f"{avg_forecast:,.0f}")

        # -------------------------
        # Forecast Table
        # -------------------------
        st.markdown("### 📋 Forecast Breakdown")

        display_df = future_df.copy()
        display_df["Forecast"] = display_df["Forecast"].round(2)
        display_df["Lower"] = display_df["Lower"].round(2)
        display_df["Upper"] = display_df["Upper"].round(2)

        st.dataframe(display_df[["Quarter", "Forecast", "Lower", "Upper"]])

        # -------------------------
        # Forecast Insight
        # -------------------------
        start_val = future_df["Forecast"].iloc[0]
        end_val = future_df["Forecast"].iloc[-1]

        growth = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0

        trend = "increasing 📈" if growth > 0 else "decreasing 📉"

        st.info(
            f"🧠 Insight: Revenue is expected to be {trend} over the next 4 quarters "
            f"with an estimated change of ~{growth:.1f}%."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TAB 5: FEATURE IMPORTANCE
    # =========================
    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Feature Importance (Drivers)")

        st.dataframe(
            importance_df.style.format({
                "Coefficient": "{:.2f}",
                "Importance": "{:.2f}"
            })
        )

        fig3 = go.Figure()

        fig3.add_trace(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation='h'
        ))

        fig3.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=350
        )


        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------
        # Top 3 Features Display (Improved UI)
        # -------------------------
        st.markdown("### Top 3 Drivers")

        top3 = importance_df.head(3).reset_index(drop=True)
        total_importance = importance_df["Importance"].sum()

        cols = st.columns(3, gap="large")

        for i, row in top3.iterrows():
            feature = row["Feature"]
            importance = row["Importance"]
            coef = row["Coefficient"]

            percent = (importance / total_importance) * 100 if total_importance != 0 else 0
            direction = "📈" if coef > 0 else "📉"
            direction_text = "Positive" if coef > 0 else "Negative"

            with cols[i]:
                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:12px;
                    background-color:#ffffff;
                    border:1px solid #e5e7eb;
                    box-shadow:0 2px 6px rgba(0,0,0,0.05);
                ">
                    <div style="font-size:14px; color:#6b7280;">Rank #{i+1}</div>
                    <div style="font-size:16px; font-weight:600; margin-bottom:6px;">{feature}</div>
                    <div style="font-size:20px; font-weight:700;">{importance:.2f}</div>
                    <div style="font-size:13px; color:#6b7280; margin-bottom:6px;">Impact Score</div>
                    <div style="font-size:13px;">{percent:.1f}% contribution</div>
                    <div style="font-size:13px; color:#374151;">{direction} {direction_text}</div>
                </div>
                """, unsafe_allow_html=True)

        # -------------------------
        # Auto Insight (final polish with % contribution)
        # -------------------------
        top_df = importance_df.head(3)

        total_importance = top_df["Importance"].sum()

        insights = []

        for _, row in top_df.iterrows():
            feature = row["Feature"]
            weight = row["Importance"]

            # Calculate percentage contribution
            percent = (weight / total_importance) * 100 if total_importance != 0 else 0

            if "revenue_lag" in feature:
                insights.append((weight, f"Seasonality drives ~{percent:.0f}% of predictions"))

            elif "revenue_yoy" in feature:
                insights.append((weight, f"Growth trends contribute ~{percent:.0f}%"))

            elif "trend" in feature:
                insights.append((weight, f"External trends influence ~{percent:.0f}%"))

            elif "liabilities" in feature:
                insights.append((weight, f"Financial obligations impact ~{percent:.0f}%"))

            elif "cash" in feature:
                insights.append((weight, f"Liquidity contributes ~{percent:.0f}%"))

        # Sort insights by importance weight (descending)
        insights = sorted(insights, key=lambda x: x[0], reverse=True)

        # Extract only text
        insight_texts = [text for _, text in insights]

        final_insight = " • ".join(insight_texts)

        st.info(f"🧠 Insight: {final_insight}")

        st.markdown('</div>', unsafe_allow_html=True)