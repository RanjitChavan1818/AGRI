export interface Probability { class: string; label: string; confidence: number }
export interface Prediction { class: string; label: string; confidence: number }
export interface GradCam { available: boolean; image: string | null }
export interface PredictionResponse { success: boolean; prediction: Prediction; probabilities: Probability[]; gradcam?: GradCam }
export interface ApiErrorResponse { success: false; error?: { code?: string; message?: string } }
export class PredictionApiError extends Error {
  readonly status?: number
  constructor(message: string, status?: number) { super(message); this.name = 'PredictionApiError'; this.status = status }
}
