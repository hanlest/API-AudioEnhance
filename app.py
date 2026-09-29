import uuid
from pathlib import Path

import gradio as gr

from resemble_enhance.audio_io import load_audio, write_mp3
from resemble_enhance.branding import PROJECT_NAME
from resemble_enhance.device import describe_device, get_inference_device
from resemble_enhance.enhancer.inference import denoise, enhance
from resemble_enhance.time_utils import fields_to_seconds, format_hms, parse_hms, split_hms_fields
from resemble_enhance.youtube_audio import download_audio_clip, fetch_video_info

device = get_inference_device()
print(f"[{PROJECT_NAME}] Inferencia en {describe_device(device)}")

EXPORT_DIR = Path(__file__).resolve().parent / ".cache" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

YOUTUBE_TAB_JS = Path(__file__).resolve().parent / "resemble_enhance" / "static" / "youtube_tab.js"
YOUTUBE_TAB_HEAD = f"<script>\n{YOUTUBE_TAB_JS.read_text(encoding='utf-8')}\n</script>"

INPUT_AUDIO_CSS = """
.input-audio-stack {
    position: relative !important;
}
.input-audio-stack .yt-source-btn-block {
    display: none !important;
}
.input-audio-stack .source-selection {
    display: flex !important;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: center;
    gap: 4px;
}
.input-audio-stack .source-selection .resemble-yt-tab {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    padding: 0;
    margin: 0;
    border: none;
    border-radius: var(--radius-md, 6px);
    background: transparent;
    color: var(--body-text-color, #fff);
    cursor: pointer;
    flex: 0 0 auto;
}
.input-audio-stack .source-selection .resemble-yt-tab:hover {
    background: var(--background-fill-secondary, rgba(255,255,255,0.08));
}
.input-audio-stack.yt-open .source-selection .resemble-yt-tab {
    background: var(--background-fill-secondary, rgba(255,255,255,0.12));
    color: #ff4444;
}
.input-audio-stack .source-selection .resemble-yt-tab svg {
    width: 1.125rem;
    height: 1.125rem;
    fill: currentColor;
    display: block;
}
.input-audio-stack.yt-open {
    min-height: 17rem;
}
.input-audio-stack .yt-inside-panel-wrap {
    position: absolute !important;
    left: 0;
    right: 0;
    top: 3.15rem;
    bottom: 0;
    z-index: 20;
    margin: 0 !important;
    padding: 0 0.75rem 0.5rem !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.input-audio-stack .yt-inside-panel-wrap > .wrap {
    gap: 0.35rem !important;
}
.input-audio-stack .yt-inside-panel-wrap .block.padded {
    padding: 0 !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-meta-block {
    min-height: 0 !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-meta-block.hide-container {
    display: none !important;
}
.input-audio-stack.yt-open .input-audio-entry > .wrap > button {
    visibility: hidden !important;
    pointer-events: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.input-audio-stack .yt-inside-panel-wrap > .wrap {
    height: 100%;
    max-height: 14rem;
    overflow-y: auto;
    padding: 0.45rem 0.65rem 0.55rem !important;
    border-radius: var(--radius-lg, 8px);
    border: 1px solid var(--border-color-primary, #374151);
    background: var(--block-background-fill, #111);
}
.input-audio-stack.yt-open .input-audio-entry .audio-container .waveform-container,
.input-audio-stack.yt-open .input-audio-entry .audio-container .controls,
.input-audio-stack.yt-open .input-audio-entry .audio-container .upload-wrap,
.input-audio-stack.yt-open .input-audio-entry .audio-container .empty,
.input-audio-stack.yt-open .input-audio-entry .audio-container .player-wrap {
    visibility: hidden !important;
    pointer-events: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-compact-url textarea {
    min-height: 2.25rem !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-time-row {
    align-items: center !important;
    gap: 0.2rem !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-time-label p {
    margin: 0 !important;
    font-size: 0.875rem;
    white-space: nowrap;
}
.input-audio-stack .yt-inside-panel-wrap .yt-time-part input {
    text-align: center !important;
    padding: 0.35rem 0.25rem !important;
    min-height: 2rem !important;
}
.input-audio-stack .yt-inside-panel-wrap .yt-time-sep p {
    margin: 0 !important;
    opacity: 0.65;
    font-weight: 600;
}
"""

OPEN_YT_JS = "() => { window.resembleOpenYoutube?.(); }"

CLOSE_YT_JS = "() => { window.resembleCloseYoutube?.(); }"


def _enhance_from_path(path: str | None, solver: str, nfe: int, tau: float, denoising: bool):
    if not path:
        raise gr.Error("No hay audio para procesar.")

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = load_audio(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1_np = wav1.cpu().numpy()
    wav2_np = wav2.cpu().numpy()

    tag = uuid.uuid4().hex[:8]
    mp3_path = EXPORT_DIR / f"enhanced_{tag}.mp3"
    try:
        write_mp3(mp3_path, wav2_np, new_sr)
    except Exception as e:
        raise gr.Error(str(e)) from e

    return (new_sr, wav1_np), (new_sr, wav2_np), str(mp3_path)


def download_youtube_only(url, url_state, start_h, start_m, start_s, end_h, end_m, end_s):
    url = (url or url_state or "").strip()
    if not url:
        raise gr.Error("Pega la URL de YouTube en el campo superior y vuelve a intentar.")
    try:
        start_sec = fields_to_seconds(start_h, start_m, start_s)
        end_sec = fields_to_seconds(end_h, end_m, end_s)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    if end_sec <= start_sec:
        raise gr.Error("El tiempo final debe ser mayor que el inicial.")
    try:
        clip_path = download_audio_clip(str(url).strip(), start_sec, end_sec)
    except Exception as e:
        raise gr.Error(str(e)) from e
    path = str(clip_path)
    clip_len = format_hms(end_sec - start_sec)
    return (
        path,
        gr.update(
            value=f"Recorte listo (**{clip_len}**). Pulsa **Mejorar audio**.",
            visible=True,
        ),
        gr.update(value=path),
        "youtube",
        gr.update(visible=False),
    )


def run_enhance(input_mode, file_path, yt_clip_path, solver, nfe, tau, denoising):
    path = yt_clip_path or file_path
    if not path:
        raise gr.Error("No hay audio para procesar. Sube un archivo o descarga un recorte de YouTube.")
    return _enhance_from_path(path, solver, nfe, tau, denoising)


def load_youtube_info(url):
    if not url or not str(url).strip():
        raise gr.Error("Ingresa una URL de YouTube.")
    try:
        title, duration = fetch_video_info(str(url).strip())
    except Exception as e:
        raise gr.Error(str(e)) from e
    end_default = min(60.0, duration)
    sh, sm, ss = split_hms_fields(0)
    eh, em, es = split_hms_fields(end_default)
    return (
        gr.update(value=f"**{title}** — duración **{format_hms(duration)}**", visible=True),
        sh,
        sm,
        ss,
        eh,
        em,
        es,
        duration,
    )


def open_youtube_panel():
    return "youtube", gr.update(visible=True)


def back_to_file_upload():
    return "file", None, gr.update(visible=False)


def on_file_selected(path, input_mode):
    if path and input_mode == "youtube":
        return gr.skip(), gr.skip(), gr.skip()
    if path:
        return "file", gr.skip(), gr.update(visible=False)
    return gr.skip(), gr.skip(), gr.skip()


def sync_yt_url(url: str):
    return url or ""


def main():
    description = (
        f"Mejora de voz con IA ({PROJECT_NAME}).\n\n"
        f"**Dispositivo:** {describe_device(device)}"
    )

    with gr.Blocks(title=PROJECT_NAME) as demo:
        gr.Markdown(f"# {PROJECT_NAME}\n{description}")

        input_mode = gr.State("file")
        yt_duration_state = gr.State(0.0)
        yt_clip_state = gr.State(None)
        yt_url_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_id="input-audio-stack", elem_classes="input-audio-stack"):
                    file_in = gr.Audio(
                        type="filepath",
                        label="Entrada de audio",
                        sources=["upload", "microphone"],
                        elem_classes="input-audio-entry",
                    )
                    yt_source_btn = gr.Button(
                        "YouTube",
                        elem_id="yt-source-trigger",
                        elem_classes=["yt-source-btn-block"],
                    )
                    with gr.Column(visible=False, elem_classes=["yt-inside-panel-wrap"]) as yt_panel:
                        yt_url = gr.Textbox(
                            show_label=False,
                            placeholder="URL de YouTube (https://...)",
                            elem_classes=["yt-compact-url"],
                            max_lines=1,
                        )
                        with gr.Row():
                            yt_info_btn = gr.Button("Info", size="sm")
                            yt_download_btn = gr.Button("Descargar recorte", size="sm", variant="primary")
                        yt_meta = gr.Markdown(visible=False, elem_classes=["yt-meta-block"])
                        with gr.Row(elem_classes=["yt-time-row"]):
                            gr.Markdown("Inicio", elem_classes=["yt-time-label"])
                            yt_start_h = gr.Textbox(
                                value="00",
                                show_label=False,
                                placeholder="HH",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=52,
                            )
                            gr.Markdown(":", elem_classes=["yt-time-sep"])
                            yt_start_m = gr.Textbox(
                                value="00",
                                show_label=False,
                                placeholder="MM",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=52,
                            )
                            gr.Markdown(":", elem_classes=["yt-time-sep"])
                            yt_start_s = gr.Textbox(
                                value="00",
                                show_label=False,
                                placeholder="SS",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=60,
                            )
                        with gr.Row(elem_classes=["yt-time-row"]):
                            gr.Markdown("Fin", elem_classes=["yt-time-label"])
                            yt_end_h = gr.Textbox(
                                value="00",
                                show_label=False,
                                placeholder="HH",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=52,
                            )
                            gr.Markdown(":", elem_classes=["yt-time-sep"])
                            yt_end_m = gr.Textbox(
                                value="01",
                                show_label=False,
                                placeholder="MM",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=52,
                            )
                            gr.Markdown(":", elem_classes=["yt-time-sep"])
                            yt_end_s = gr.Textbox(
                                value="00",
                                show_label=False,
                                placeholder="SS",
                                max_lines=1,
                                elem_classes=["yt-time-part"],
                                scale=0,
                                min_width=60,
                            )

                    close_yt = gr.Button("cerrar yt", visible=False, elem_id="close-yt-mode")

                solver = gr.Dropdown(
                    choices=["Midpoint", "RK4", "Euler"],
                    value="Midpoint",
                    label="Solucionador ODE (CFM)",
                )
                nfe = gr.Slider(1, 128, value=64, step=1, label="Evaluaciones de la función (NFE)")
                tau = gr.Slider(0, 1, value=0.5, step=0.01, label="Temperatura previa (CFM)")
                denoising = gr.Checkbox(value=False, label="Priorizar denoise antes de mejorar")
                run_btn = gr.Button("Mejorar audio", variant="primary")

            with gr.Column(scale=1):
                out_denoised = gr.Audio(label="Salida: sin ruido")
                out_enhanced = gr.Audio(label="Salida: mejorada")
                mp3_download = gr.DownloadButton(
                    "Descargar audio mejorado (MP3)",
                    value=None,
                    variant="secondary",
                )

        yt_source_btn.click(
            open_youtube_panel,
            outputs=[input_mode, yt_panel],
            js=OPEN_YT_JS,
        )

        close_yt.click(
            back_to_file_upload,
            outputs=[input_mode, yt_clip_state, yt_panel],
            js=CLOSE_YT_JS,
        )

        file_in.change(
            on_file_selected,
            inputs=[file_in, input_mode],
            outputs=[input_mode, yt_clip_state, yt_panel],
        )

        yt_url.change(sync_yt_url, inputs=[yt_url], outputs=[yt_url_state])
        yt_url.input(sync_yt_url, inputs=[yt_url], outputs=[yt_url_state])

        yt_info_btn.click(
            load_youtube_info,
            inputs=[yt_url],
            outputs=[
                yt_meta,
                yt_start_h,
                yt_start_m,
                yt_start_s,
                yt_end_h,
                yt_end_m,
                yt_end_s,
                yt_duration_state,
            ],
        )

        yt_download_btn.click(
            download_youtube_only,
            inputs=[
                yt_url,
                yt_url_state,
                yt_start_h,
                yt_start_m,
                yt_start_s,
                yt_end_h,
                yt_end_m,
                yt_end_s,
            ],
            outputs=[yt_clip_state, yt_meta, file_in, input_mode, yt_panel],
        ).then(None, None, None, js=CLOSE_YT_JS)

        run_btn.click(
            run_enhance,
            inputs=[input_mode, file_in, yt_clip_state, solver, nfe, tau, denoising],
            outputs=[out_denoised, out_enhanced, mp3_download],
        )

    demo.launch(css=INPUT_AUDIO_CSS, head=YOUTUBE_TAB_HEAD)


if __name__ == "__main__":
    main()
