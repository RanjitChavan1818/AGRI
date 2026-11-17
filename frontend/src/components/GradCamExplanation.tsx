import type { GradCam } from '../types/prediction'

interface Props { gradcam?: GradCam }

export default function GradCamExplanation({ gradcam }: Props) {
  const image = gradcam?.available === true && typeof gradcam.image === 'string' ? gradcam.image : null

  return <section className="mt-7 border-t border-emerald-950/10 pt-7" aria-labelledby="ai-explanation-title">
    <h3 id="ai-explanation-title" className="font-extrabold text-ink">AI Explanation</h3>
    <p className="mt-1 text-sm leading-6 text-slate-500">Highlighted regions show the areas that most influenced the model&apos;s prediction.</p>
    {image ? <>
      <div className="mt-4 overflow-hidden rounded-2xl border border-emerald-950/10 bg-forest-50 shadow-[0_10px_25px_-20px_rgba(21,57,35,.6)]">
        <img src={image} alt="Grad-CAM heatmap showing regions that influenced the corn leaf prediction" className="block h-auto max-h-80 w-full object-contain" />
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-600">Warmer regions indicate areas that contributed more strongly to the model&apos;s prediction.</p>
      <p className="mt-2 text-xs leading-5 text-slate-500">Grad-CAM shows which image regions influenced the model; it does not guarantee that the outlined region is the actual cause of the disease.</p>
    </> : <p className="mt-4 rounded-xl border border-forest-700/10 bg-forest-50 px-4 py-3 text-sm leading-6 text-slate-600">AI explanation is temporarily unavailable, but the prediction result is still available.</p>}
  </section>
}
