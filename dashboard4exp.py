import sys
import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Toplevel, LabelFrame
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

# Optional: For LSTM deep learning model
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

class SentimentStockApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Social Media Sentiment and Stock Market Dashboard")
        self.geometry("1200x1050")  # increased height for all features including new prescriptive button
        self.social_data = None
        self.stock_data = None
        self.current_figure = None  # Store current matplotlib figure for saving

        self.create_widgets()

    def create_widgets(self):
        # Buttons for file upload
        frame = tk.Frame(self)
        frame.pack(pady=20)

        self.btn_social = tk.Button(frame, text="Upload Social Media Dataset (CSV)", command=self.load_social_file)
        self.btn_social.grid(row=0, column=0, padx=20)

        self.btn_stock = tk.Button(frame, text="Upload Stock Market Dataset (CSV)", command=self.load_stock_file)
        self.btn_stock.grid(row=0, column=1, padx=20)

        self.btn_analyze = tk.Button(self, text="Perform Analysis", command=self.perform_analysis)
        self.btn_analyze.pack(pady=10)

        self.btn_download = tk.Button(self, text="Download Plots", command=self.download_plots, state=tk.DISABLED)
        self.btn_download.pack(pady=5)

        # Diagnostic Analysis button
        self.btn_diagnostic = tk.Button(self, text="Diagnostic Analysis", command=self.perform_diagnostic_analysis_thread, state=tk.DISABLED)
        self.btn_diagnostic.pack(pady=5)

        # Prediction button
        self.btn_predict = tk.Button(self, text="Predict Next-Day Price", command=self.perform_prediction_thread, state=tk.DISABLED)
        self.btn_predict.pack(pady=5)

        # Prescriptive Analysis button
        self.btn_prescriptive = tk.Button(self, text="Prescriptive Analysis", command=self.perform_prescriptive_analysis_thread, state=tk.DISABLED)
        self.btn_prescriptive.pack(pady=5)

        # Frame for plots with scroll support
        self.plot_container = tk.Frame(self)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar for scrolling plots
        self.canvas = tk.Canvas(self.plot_container)
        self.scrollbar = tk.Scrollbar(self.plot_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Inner frame inside canvas to hold plots
        self.plot_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.plot_frame, anchor='nw')

    def load_social_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filename:
            return
        try:
            df = pd.read_csv(filename)
            if 'text' not in df.columns or 'date' not in df.columns:
                raise ValueError("Social media CSV must include 'text' and 'date' columns")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'text'])
            self.social_data = df
            messagebox.showinfo("Success", f"Loaded social media data with {len(df)} records")
            self._update_buttons_state()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load social media data:\n{str(e)}")

    def load_stock_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filename:
            return
        try:
            df = pd.read_csv(filename)
            # Strip whitespace and convert to lowercase for columns
            df.columns = df.columns.str.strip().str.lower()

            if 'date' not in df.columns or 'close' not in df.columns:
                raise ValueError("Stock CSV must include 'date' and 'close' columns")
        
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Optional volume column
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = None

            df = df.dropna(subset=['date', 'close'])
            self.stock_data = df.sort_values('date')
            messagebox.showinfo("Success", f"Loaded stock data with {len(df)} records")
            self._update_buttons_state()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load stock data:\n{str(e)}")

    def _update_buttons_state(self):
        # Enable diagnostic, prediction, and prescriptive if both datasets loaded
        if self.social_data is not None and self.stock_data is not None:
            self.btn_diagnostic.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)
            self.btn_prescriptive.config(state=tk.NORMAL)
        else:
            self.btn_diagnostic.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
            self.btn_prescriptive.config(state=tk.DISABLED)

    def perform_analysis(self):
        if self.social_data is None or self.stock_data is None:
            messagebox.showwarning("Missing Data", "Please upload both social media and stock market datasets before analysis.")
            return
        threading.Thread(target=self._analyze_and_plot).start()

    def _analyze_and_plot(self):
        try:
            self.btn_analyze.config(state=tk.DISABLED)
            self.btn_download.config(state=tk.DISABLED)
            self.btn_diagnostic.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
            self.btn_prescriptive.config(state=tk.DISABLED)

            # Compute sentiment scores
            self.social_data['sentiment_score'] = self.social_data['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            self.social_data['sentiment_category'] = self.social_data['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
            
            sentiment_counts = self.social_data['sentiment_category'].value_counts()
            sentiment_trend = self.social_data.groupby(self.social_data['date'].dt.date)['sentiment_score'].mean()
            stock_data = self.stock_data.copy()
            stock_data['date'] = stock_data['date'].dt.date
            merged = pd.merge(sentiment_trend.reset_index(name='avg_sentiment'), stock_data[['date', 'close', 'volume']], on='date', how='inner')
            merged['norm_close'] = (merged['close'] - merged['close'].min()) / (merged['close'].max() - merged['close'].min())

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig, axes = plt.subplots(5, 1, figsize=(12, 18))
            fig.tight_layout(pad=5.0)

            axes[0].pie(
                [sentiment_counts.get('positive',0), sentiment_counts.get('neutral',0), sentiment_counts.get('negative',0)],
                labels=['Positive', 'Neutral', 'Negative'],
                colors=['#22eeaa', '#8888aa', '#ff5566'],
                autopct='%1.1f%%',
                startangle=140
            )
            axes[0].set_title('Sentiment Distribution')

            axes[1].plot(sentiment_trend.index, sentiment_trend.values, color='#22eeaa', marker='o')
            axes[1].set_title('Sentiment Trend Over Time')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Average Sentiment Score')
            axes[1].tick_params(axis='x', rotation=45)

            axes[2].plot(stock_data['date'], stock_data['close'], color='#00d9ff', marker='o')
            axes[2].set_title('Stock Price Over Time')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Close Price')
            axes[2].tick_params(axis='x', rotation=45)

            axes[3].plot(merged['date'], merged['avg_sentiment'], color='#22eeaa', label='Avg Sentiment')
            axes[3].plot(merged['date'], merged['norm_close'], color='#00d9ff', label='Normalized Stock Close')
            axes[3].set_title('Correlation: Avg Sentiment vs Normalized Stock Price')
            axes[3].legend()
            axes[3].set_xlabel('Date')
            axes[3].tick_params(axis='x', rotation=45)

            vol_sent_data = merged.dropna(subset=['volume', 'avg_sentiment'])
            if not vol_sent_data.empty:
                axes[4].scatter(vol_sent_data['volume'], vol_sent_data['avg_sentiment'], color='#ff9900', alpha=0.7)
                axes[4].set_title('Volume vs Average Sentiment')
                axes[4].set_xlabel('Volume (Traded Shares)')
                axes[4].set_ylabel('Average Sentiment Score')
                axes[4].set_xscale('log')
                axes[4].grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
            else:
                axes[4].text(0.5, 0.5, 'Volume data not available to plot', ha='center', va='center', fontsize=12)
                axes[4].set_axis_off()

            self.current_figure = fig
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.btn_download.config(state=tk.NORMAL)
            self.btn_diagnostic.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)
            self.btn_prescriptive.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}")
            self.current_figure = None
            self.btn_download.config(state=tk.DISABLED)
            self.btn_diagnostic.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
            self.btn_prescriptive.config(state=tk.DISABLED)
        finally:
            self.btn_analyze.config(state=tk.NORMAL)

    def download_plots(self):
        if self.current_figure is None:
            messagebox.showwarning("No Plot", "Please perform analysis first to generate plots before downloading.")
            return
        try:
            filetypes = [('PNG Image', '*.png'), ('JPEG Image', '*.jpg'), ('All files', '*.*')]
            filename = filedialog.asksaveasfilename(
                title="Save current plots as image",
                defaultextension=".png",
                filetypes=filetypes
            )
            if not filename:
                return

            self.current_figure.savefig(filename)
            messagebox.showinfo("Saved", f"Plots image saved successfully:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plots image:\n{str(e)}")

    def perform_diagnostic_analysis_thread(self):
        threading.Thread(target=self.perform_diagnostic_analysis).start()

    def perform_diagnostic_analysis(self):
        self.btn_diagnostic.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_prescriptive.config(state=tk.DISABLED)
        try:
            if self.social_data is None or self.stock_data is None:
                messagebox.showwarning("Missing Data", "Please upload both datasets to perform diagnostic analysis.")
                return

            self.social_data['sentiment_score'] = self.social_data['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            sentiment_trend = self.social_data.groupby(self.social_data['date'].dt.date)['sentiment_score'].mean()
            stock_data = self.stock_data.copy()
            stock_data['date'] = stock_data['date'].dt.date
            merged = pd.merge(sentiment_trend.reset_index(name='avg_sentiment'), stock_data[['date', 'close', 'volume']], on='date', how='inner')

            merged_sorted = merged.sort_values('date')
            merged_sorted['price_pct_change'] = merged_sorted['close'].pct_change()
            correlation = merged_sorted['avg_sentiment'].corr(merged_sorted['price_pct_change'])

            high_sent_threshold = merged_sorted['avg_sentiment'].quantile(0.75)
            merged_sorted['high_sentiment'] = merged_sorted['avg_sentiment'] >= high_sent_threshold
            vol_high = merged_sorted[merged_sorted['high_sentiment']]['price_pct_change'].std()
            vol_other = merged_sorted[~merged_sorted['high_sentiment']]['price_pct_change'].std()

            top_vol_dates = merged_sorted.nlargest(3, 'volume')['date'].tolist()

            event_win = Toplevel(self)
            event_win.title("Diagnostic Analysis Results")
            event_win.geometry("1000x900")

            text_frame = LabelFrame(event_win, text="Diagnostic Summary", padx=10, pady=10)
            text_frame.pack(fill=tk.X, padx=10, pady=5)

            summary_text = (
                f"Correlation between Avg Sentiment and Daily Stock Price Change: {correlation:.4f}\n\n"
                f"Stock Price Volatility Std Dev on High Sentiment Days: {vol_high:.6f}\n"
                f"Stock Price Volatility Std Dev on Other Days: {vol_other:.6f}\n\n"
                "Top 3 Volume Days for Event Impact Analysis:\n" +
                "\n".join(str(d) for d in top_vol_dates)
            )
            st = scrolledtext.ScrolledText(text_frame, height=10, wrap=tk.WORD)
            st.pack(fill=tk.X)
            st.insert(tk.END, summary_text)
            st.configure(state='disabled')

            plot_frame = tk.Frame(event_win)
            plot_frame.pack(fill=tk.BOTH, expand=True)

            fig, axes = plt.subplots(len(top_vol_dates), 1, figsize=(10, 4 * len(top_vol_dates)), sharex=True)
            if len(top_vol_dates) == 1:
                axes = [axes]

            for ax, event_date in zip(axes, top_vol_dates):
                mask = (merged_sorted['date'] >= event_date - pd.Timedelta(days=5)) & (merged_sorted['date'] <= event_date + pd.Timedelta(days=5))
                window_data = merged_sorted[mask]
                ax.plot(window_data['date'], window_data['close'], label='Close Price', color='blue', marker='o')
                ax.set_ylabel('Close Price', color='blue')
                ax2 = ax.twinx()
                ax2.plot(window_data['date'], window_data['avg_sentiment'], label='Avg Sentiment', color='green', marker='x')
                ax2.set_ylabel('Avg Sentiment', color='green')
                ax.set_title(f"Event Impact Window ±5 days around {event_date}")
                ax.tick_params(axis='x', rotation=45)

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Diagnostic Error", f"An error occurred during diagnostic analysis:\n{str(e)}")
        finally:
            self.btn_diagnostic.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)
            self.btn_prescriptive.config(state=tk.NORMAL)

    def perform_prediction_thread(self):
        threading.Thread(target=self.perform_prediction).start()

    def perform_prediction(self):
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_diagnostic.config(state=tk.DISABLED)
        self.btn_prescriptive.config(state=tk.DISABLED)
        try:
            if self.social_data is None or self.stock_data is None:
                messagebox.showwarning("Missing Data", "Please upload both datasets before prediction.")
                return

            self.social_data['sentiment_score'] = self.social_data['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            sentiment_trend = self.social_data.groupby(self.social_data['date'].dt.date)['sentiment_score'].mean()

            stock_data = self.stock_data.copy()
            stock_data['date'] = stock_data['date'].dt.date

            # Merge close price and sentiment
            data = pd.merge(stock_data[['date', 'close']], sentiment_trend.reset_index(name='avg_sentiment'), on='date', how='inner')
            data = data.sort_values('date')

            # Compute next day price movement
            data['next_close'] = data['close'].shift(-1)
            data.dropna(inplace=True)
            data['price_diff'] = data['next_close'] - data['close']

            # Features and target
            X = data[['close', 'avg_sentiment']]
            y = data['price_diff']

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            mse_lr = mean_squared_error(y_test, y_pred_lr)

            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred_rf)

            # LSTM model if available
            if TF_AVAILABLE:
                scaler = MinMaxScaler()
                scaled_X = scaler.fit_transform(X)
                scaled_y = y.values.reshape(-1, 1)
                scaled_y = scaler.fit_transform(scaled_y)

                seq_len = 3  # sequence length
                def create_sequences(X_in, y_in, seq_length):
                    Xs, ys = [], []
                    for i in range(len(X_in) - seq_length):
                        Xs.append(X_in[i:(i+seq_length)])
                        ys.append(y_in[i+seq_length])
                    return np.array(Xs), np.array(ys)

                X_seq, y_seq = create_sequences(scaled_X, scaled_y, seq_len)
                train_size = int(len(X_seq)*0.8)
                X_train_lstm, X_test_lstm = X_seq[:train_size], X_seq[train_size:]
                y_train_lstm, y_test_lstm = y_seq[:train_size], y_seq[train_size:]

                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(seq_len, X.shape[1])))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')

                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=8, validation_split=0.1, callbacks=[early_stop], verbose=0)

                y_pred_lstm = model.predict(X_test_lstm).flatten()
                y_test_inv = scaler.inverse_transform(y_test_lstm)
                y_pred_inv = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1))
                mse_lstm = mean_squared_error(y_test_inv, y_pred_inv)

                # Matching lengths for averaging predictions
                y_pred_lr_trunc = y_pred_lr[-len(y_pred_inv):]
                y_pred_rf_trunc = y_pred_rf[-len(y_pred_inv):]
                avg_pred_diff = (y_pred_lr_trunc + y_pred_rf_trunc + y_pred_inv.flatten()) / 3

                close_prices_to_use = X_test[-len(y_pred_inv):]['close'].values
                test_dates = data['date'].iloc[-len(y_pred_inv):].values

                y_test_to_use = y_test[-len(y_pred_inv):].values  # To plot actual for same length
            else:
                mse_lstm = None
                avg_pred_diff = (y_pred_lr + y_pred_rf) / 2
                close_prices_to_use = X_test['close'].values
                test_dates = data['date'].iloc[-len(X_test):].values
                y_test_to_use = y_test.values

            # Compute predicted next-day prices = today close price + average predicted diff
            predicted_next_prices = close_prices_to_use + avg_pred_diff

            # Plot predictions vs actual for test set
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test_to_use, label='Actual', marker='o')
            ax.plot(y_pred_lr[-len(avg_pred_diff):], label=f'Linear Regression (MSE={mse_lr:.4f})', marker='x')
            ax.plot(y_pred_rf[-len(avg_pred_diff):], label=f'Random Forest (MSE={mse_rf:.4f})', marker='^')
            if mse_lstm is not None:
                ax.plot(y_pred_inv.flatten(), label=f'LSTM (MSE={mse_lstm:.4f})', marker='s')
            ax.plot(avg_pred_diff, label='Average Predicted Diff', linestyle='--', color='black')
            ax.set_title('Next-Day Price Movement Prediction')
            ax.set_xlabel('Test Sample Index')
            ax.set_ylabel('Price Movement (Next Day Close - Close)')
            ax.legend()
            ax.grid(True)

            # Show plot in new window
            pred_win = Toplevel(self)
            pred_win.title("Prediction Results")
            pred_win.geometry("1100x900")

            canvas = FigureCanvasTkAgg(fig, master=pred_win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Display predicted next-day prices for first 10 samples below plot
            text_frame = LabelFrame(pred_win, text="Average Predicted Next-Day Prices (first 10 samples)", padx=10, pady=10)
            text_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=False)

            st = scrolledtext.ScrolledText(text_frame, height=10, wrap=tk.WORD)
            st.pack(fill=tk.BOTH, expand=True)

            lines = ["Date       | Predicted Next-Day Close Price"]
            lines.append("-"*36)
            for i in range(min(10, len(predicted_next_prices))):
                date_str = pd.to_datetime(test_dates[i]).strftime("%Y-%m-%d")
                price_str = f"{predicted_next_prices[i]:.4f}"
                lines.append(f"{date_str} | {price_str}")

            st.insert(tk.END, "\n".join(lines))
            st.configure(state='disabled')

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")
        finally:
            self.btn_predict.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_diagnostic.config(state=tk.NORMAL)
            self.btn_prescriptive.config(state=tk.NORMAL)

    def perform_prescriptive_analysis_thread(self):
        threading.Thread(target=self.perform_prescriptive_analysis).start()

    def perform_prescriptive_analysis(self):
        self.btn_prescriptive.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_diagnostic.config(state=tk.DISABLED)
        try:
            if self.social_data is None or self.stock_data is None:
                messagebox.showwarning("Missing Data", "Please upload both datasets to perform prescriptive analysis.")
                return

            # Compute sentiment score average per day
            self.social_data['sentiment_score'] = self.social_data['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            sentiment_trend = self.social_data.groupby(self.social_data['date'].dt.date)['sentiment_score'].mean()
            stock_data = self.stock_data.copy()
            stock_data['date'] = stock_data['date'].dt.date

            merged = pd.merge(sentiment_trend.reset_index(name='avg_sentiment'), stock_data[['date', 'close']], on='date', how='inner')
            merged = merged.sort_values('date')

            # Calculate price daily change
            merged['price_diff'] = merged['close'].diff()
            merged.dropna(inplace=True)

            # Generate Buy signal: Buy when sentiment > 0.5 and price drops (price_diff < 0)
            merged['buy_signal'] = (merged['avg_sentiment'] > 0.5) & (merged['price_diff'] < 0)

            # Marketing insight recommendation:
            # If negative sentiment correlates strongly with price drop, suggest timing announcements when sentiment low

            # Calculate correlation between sentiment and price_diff on negative sentiment days
            negative_sentiment = merged[merged['avg_sentiment'] < 0]
            if not negative_sentiment.empty:
                corr_neg = negative_sentiment['avg_sentiment'].corr(negative_sentiment['price_diff'])
            else:
                corr_neg = None

            marketing_recommendation = "Insufficient negative sentiment days data for marketing recommendation."
            if corr_neg is not None and corr_neg < -0.3:
                marketing_recommendation = (
                    "Negative sentiment shows moderate to strong correlation with price drops.\n"
                    "Suggest timing product announcements or PR campaigns during periods of positive or improving sentiment."
                )
            else:
                marketing_recommendation = (
                    "No strong negative sentiment correlation with price drops detected.\n"
                    "Marketing timing based on sentiment may not be critical."
                )

            # Count buy signals
            total_signals = merged['buy_signal'].sum()

            # Prepare textual summary
            summary_lines = [
                f"Total 'Buy' signals detected (Sentiment > 0.5 & Price Drop): {total_signals}",
                "",
                "Sample 'Buy' signal dates:",
            ]
            buy_dates = merged[merged['buy_signal']]['date'].head(10).tolist()
            if buy_dates:
                summary_lines.extend([str(d) for d in buy_dates])
            else:
                summary_lines.append("No buy signal dates.")

            summary_lines.append("")
            summary_lines.append("Marketing Insight Recommendation:")
            summary_lines.append(marketing_recommendation)

            # Show results in a new window
            presc_win = Toplevel(self)
            presc_win.title("Prescriptive Analysis Results")
            presc_win.geometry("700x600")

            label_frame = LabelFrame(presc_win, text="Prescriptive Analysis Summary", padx=10, pady=10)
            label_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            st = scrolledtext.ScrolledText(label_frame, wrap=tk.WORD)
            st.pack(fill=tk.BOTH, expand=True)
            st.insert(tk.END, "\n".join(summary_lines))
            st.configure(state='disabled')

        except Exception as e:
            messagebox.showerror("Prescriptive Analysis Error", f"An error occurred during prescriptive analysis:\n{str(e)}")
        finally:
            self.btn_prescriptive.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)
            self.btn_diagnostic.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = SentimentStockApp()
    app.mainloop()

