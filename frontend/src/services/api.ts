import type { ApiErrorResponse, GradCam, Prediction, PredictionResponse, Probability } from '../types/prediction'
import { PredictionApiError } from '../types/prediction'
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL
function isPrediction(value: unknown): value is Prediction {
  if (!value || typeof value !== 'object') return false
  const prediction = value as Record<string, unknown>
  return typeof prediction.class === 'string' && typeof prediction.label === 'string' && typeof prediction.confidence === 'number'
}

function isProbability(value: unknown): value is Probability {
  return isPrediction(value)
}

function isGradCam(value: unknown): value is GradCam {
  if (!value || typeof value !== 'object') return false
  const gradcam = value as Record<string, unknown>
  return typeof gradcam.available === 'boolean' && (typeof gradcam.image === 'string' || gradcam.image === null)
}

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== 'object') return false
  const response = value as Record<string, unknown>
  return response.success === true && isPrediction(response.prediction) && Array.isArray(response.probabilities) && response.probabilities.length === 4 && response.probabilities.every(isProbability) && (!('gradcam' in response) || isGradCam(response.gradcam))
}
export async function predictCornDisease(file: File): Promise<PredictionResponse> {
  if (!apiBaseUrl) throw new PredictionApiError('The prediction service is not configured.')
  const formData = new FormData(); formData.append('image', file)
  let response: Response
  try { response = await fetch(`${apiBaseUrl}/predict/`, { method: 'POST', body: formData }) } catch { throw new PredictionApiError('Unable to reach the prediction service. Please check your connection and try again.') }
  let payload: unknown; try { payload = await response.json() } catch { throw new PredictionApiError('The prediction service returned an invalid response.', response.status) }
  if (!response.ok) { const error = payload as ApiErrorResponse; if (response.status === 413) throw new PredictionApiError('This image is too large. Please choose an image smaller than 5 MB.', 413); if (response.status >= 500) throw new PredictionApiError('Unable to analyze the image right now. Please try again.', response.status); throw new PredictionApiError(error.error?.message || 'Please upload a valid JPG, PNG, or WEBP image.', response.status) }
  if (!isPredictionResponse(payload)) throw new PredictionApiError('The prediction service returned an invalid response.', response.status)
  return payload
}
