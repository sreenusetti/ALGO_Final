# cpr_ai_predictor.py (Final, Cleaned)
import joblib
import numpy as np
import logging

#logging.basicConfig(level=logging.DEBUG)

class CPR_AIPredictor:
    def __init__(self, model_path="ai_cpr_model.pkl", order_manager=None, logger=None):
        self.order_manager = order_manager
        self.logger = logger or logging.getLogger(__name__)
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f"[AI-CPR] Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"[AI-CPR] Failed to load model: {e}")
            self.model = None

    def predict(self, indicators, pivot_data, feature_builder):
        """
        Make AI prediction using the loaded model.
        Returns tuple: (label, confidence, distribution, extra_info)
        """
        if self.model is None:
            self.logger.warning("[AI-CPR] No model loaded. Returning NEUTRAL.")
            return "NEUTRAL", 0.0, None, "AI model not active"

        try:
            # Build features using the provided feature_builder
            raw_features = feature_builder(None, indicators, pivot_data)

            # Normalize to 2D numpy array
            features = np.array(raw_features, dtype=float)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.ndim != 2:
                self.logger.error(f"[AI-CPR] Invalid feature shape: {features.shape}")
                return "HOLD", 0.0, None, "Invalid features"

            self.logger.debug(f"[AI-CPR] Features: {features.tolist()}")

            # Make prediction
            pred = self.model.predict(features)
            action = pred[0] if len(pred) > 0 else "HOLD"

            # Assuming the model doesn't provide explicit confidence;
            # for now, use a default confidence
            confidence = 0.7 if action != "HOLD" else 0.0

            # For distribution, assume None for now
            distribution = None

            self.logger.debug(f"[AI-CPR] Prediction: {action}, Confidence: {confidence}")

            return action, confidence, distribution, "AI prediction successful"

        except Exception as e:
            self.logger.error(f"[AI-CPR] Prediction error: {e}", exc_info=True)
            return "HOLD", 0.0, None, f"Prediction error: {str(e)}"

    def trade(self, ltp=None, indicators=None, pivot_data=None,
              feature_builder=None, symbol=None, trade_qty=1, place_order_fn=None):
        """
        Main AI trading logic.
        - feature_builder must accept (ltp, indicators, pivot_data) and return np.ndarray shape (1, N)
        - place_order_fn is an optional callback: fn(action) -> bool/response
        """
        if self.model is None:
            self.logger.error("[AI-CPR] No model loaded. Skipping trade.")
            return None

        try:
            if not feature_builder:
                self.logger.error("[AI-CPR] No feature_builder provided.")
                return None

            raw = feature_builder(ltp, indicators, pivot_data)

            # Normalize features to numpy 2D
            features = np.array(raw, dtype=float)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.ndim != 2:
                self.logger.error(f"[AI-CPR] Invalid feature shape: {getattr(features, 'shape', 'unknown')}")
                return None

            self.logger.debug(f"[AI-CPR] Features for {symbol}: {features.tolist()}")

            # prediction
            pred = self.model.predict(features)
            action = pred[0] if len(pred) > 0 else "HOLD"
            self.logger.info(f"[AI-CPR] Prediction for {symbol}: {action}")

            # If callback provided, use it (symbol-specific), else fallback to order_manager
            place_result = None
            if place_order_fn:
                try:
                    self.logger.debug(f"[AI-CPR] Calling place_order_fn for {symbol} with action={action}")
                    place_result = place_order_fn(action, symbol)
                    self.logger.debug(f"[AI-CPR] place_order_fn returned: {place_result}")
                except Exception as ex:
                    self.logger.error(f"[AI-CPR] place_order_fn raised exception for {symbol}: {ex}", exc_info=True)
            elif self.order_manager:
                try:
                    if action == "BUY":
                        self.order_manager.ai_exit_all(symbol)
                        place_result = self.order_manager.ai_buy(symbol, qty=trade_qty)
                    elif action == "SELL":
                        self.order_manager.ai_exit_all(symbol)
                        place_result = self.order_manager.ai_sell(symbol, qty=trade_qty)
                    else:
                        self.logger.info(f"[AI-CPR] HOLD: No AI action taken for {symbol}")
                except Exception as ex:
                    self.logger.error(f"[AI-CPR] OrderManager action failed for {symbol}: {ex}", exc_info=True)
            else:
                self.logger.warning("[AI-CPR] No order manager or place_order_fn provided. No order placed.")

            # Return both for debugging: predicted action and place_result (bool or response)
            return {"action": action, "placed": place_result}

        except Exception as e:
            self.logger.error(f"[AI-CPR] Prediction error for {symbol}: {e}", exc_info=True)
            return None
