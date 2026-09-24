/**
 * useTypographyCanvas.ts
 * =======================
 * React hook driving a Konva.js overlay for WYSIWYG, style-preserving
 * inline text editing on top of a rendered PDF page / PNG image.
 *
 * Responsibilities:
 *  - Render one Konva.Text node per TypographyProfile region.
 *  - On double-click, swap the node for an absolutely-positioned HTML
 *    <textarea> (Konva has no native text input) matched to the node's
 *    font metrics.
 *  - On commit, re-measure the new string against the original bbox and
 *    live-shrink font size / letter-spacing client-side for instant visual
 *    feedback, then POST the edit to the backend for the authoritative
 *    server-side render.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Konva from "konva";

export interface TypographyRegion {
  region_id: string;
  page: number;
  bbox: [number, number, number, number]; // x0, y0, x1, y1 in PDF/image px
  baseline_y: number;
  text: string;
  font_family: string;
  font_family_fallback: string;
  font_size_pt: number;
  weight: number;
  italic: boolean;
  letter_spacing: number;
  line_height: number;
  color_rgb: [number, number, number];
  alignment: "left" | "center" | "right";
}

interface UseTypographyCanvasOptions {
  stageRef: React.RefObject<Konva.Stage>;
  layerRef: React.RefObject<Konva.Layer>;
  regions: TypographyRegion[];
  scale: number; // canvas render scale relative to source doc coordinates
  onCommitEdit: (regionId: string, newText: string) => Promise<void>;
  minFontSize?: number;
}

interface EditingState {
  regionId: string;
  value: string;
  screenX: number;
  screenY: number;
  screenWidth: number;
  screenHeight: number;
}

const rgbToCss = ([r, g, b]: [number, number, number]) => `rgb(${r}, ${g}, ${b})`;

function fontStringFor(region: TypographyRegion, fontSizePx: number): string {
  const style = region.italic ? "italic" : "normal";
  const weight = region.weight;
  const family = `${region.font_family}, ${region.font_family_fallback}`;
  return `${style} ${weight} ${fontSizePx}px ${family}`;
}

/** Binary-search shrink so `text` measured in `ctx` fits within `maxWidth`. */
function fitFontSize(
  ctx: CanvasRenderingContext2D,
  text: string,
  baseFont: (size: number) => string,
  startSize: number,
  maxWidth: number,
  minSize: number
): number {
  ctx.font = baseFont(startSize);
  if (ctx.measureText(text).width <= maxWidth) return startSize;

  let lo = minSize;
  let hi = startSize;
  for (let i = 0; i < 30 && hi - lo > 0.25; i++) {
    const mid = (lo + hi) / 2;
    ctx.font = baseFont(mid);
    if (ctx.measureText(text).width <= maxWidth) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

export function useTypographyCanvas({
  stageRef,
  layerRef,
  regions,
  scale,
  onCommitEdit,
  minFontSize = 6,
}: UseTypographyCanvasOptions) {
  const nodesRef = useRef<Map<string, Konva.Text>>(new Map());
  const [editing, setEditing] = useState<EditingState | null>(null);
  const measureCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    const canvas = document.createElement("canvas");
    measureCtxRef.current = canvas.getContext("2d");
  }, []);

  // ---- Render / sync Konva.Text nodes from region data ----
  const syncNodes = useCallback(() => {
    const layer = layerRef.current;
    if (!layer) return;

    const seen = new Set<string>();

    for (const region of regions) {
      seen.add(region.region_id);
      const [x0, y0, x1] = region.bbox;
      const fontSizePx = region.font_size_pt * (96 / 72) * scale; // pt -> px

      let node = nodesRef.current.get(region.region_id);
      if (!node) {
        node = new Konva.Text({
          id: region.region_id,
          x: x0 * scale,
          y: y0 * scale,
          text: region.text,
          fontSize: fontSizePx,
          fontFamily: `${region.font_family}, ${region.font_family_fallback}`,
          fontStyle: `${region.italic ? "italic" : "normal"} ${region.weight}`,
          fill: rgbToCss(region.color_rgb),
          letterSpacing: region.letter_spacing * scale,
          align: region.alignment,
          width: (x1 - x0) * scale,
          listening: true,
        });
        node.on("dblclick dbltap", () => beginEdit(region, node!));
        layer.add(node);
        nodesRef.current.set(region.region_id, node);
      } else {
        node.setAttrs({
          x: x0 * scale,
          y: y0 * scale,
          text: region.text,
          fontSize: fontSizePx,
          fill: rgbToCss(region.color_rgb),
        });
      }
    }

    // Remove nodes for regions no longer present.
    for (const [id, node] of nodesRef.current.entries()) {
      if (!seen.has(id)) {
        node.destroy();
        nodesRef.current.delete(id);
      }
    }

    layer.batchDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regions, scale]);

  useEffect(syncNodes, [syncNodes]);

  // ---- Begin inline edit: overlay a matched <textarea> ----
  const beginEdit = useCallback(
    (region: TypographyRegion, node: Konva.Text) => {
      const stage = stageRef.current;
      if (!stage) return;

      const box = node.getClientRect({ relativeTo: stage as unknown as Konva.Container });
      const containerRect = stage.container().getBoundingClientRect();

      setEditing({
        regionId: region.region_id,
        value: region.text,
        screenX: containerRect.left + box.x,
        screenY: containerRect.top + box.y,
        screenWidth: box.width,
        screenHeight: box.height,
      });

      node.hide();
      node.getLayer()?.batchDraw();
    },
    [stageRef]
  );

  // ---- Live-preview font fitting as the user types ----
  const previewFit = useCallback(
    (region: TypographyRegion, node: Konva.Text, value: string) => {
      const ctx = measureCtxRef.current;
      if (!ctx) return;

      const startSizePx = region.font_size_pt * (96 / 72) * scale;
      const boxWidthPx = (region.bbox[2] - region.bbox[0]) * scale;

      const fitted = fitFontSize(
        ctx,
        value,
        (size) => fontStringFor(region, size),
        startSizePx,
        boxWidthPx,
        minFontSize
      );

      node.text(value);
      node.fontSize(fitted);
    },
    [scale, minFontSize]
  );

  // ---- Commit edit: restore node, push authoritative render request ----
  const commitEdit = useCallback(async () => {
    if (!editing) return;
    const { regionId, value } = editing;
    const node = nodesRef.current.get(regionId);

    setEditing(null);
    if (node) {
      node.show();
      node.getLayer()?.batchDraw();
    }

    await onCommitEdit(regionId, value);
  }, [editing, onCommitEdit]);

  const cancelEdit = useCallback(() => {
    if (!editing) return;
    const node = nodesRef.current.get(editing.regionId);
    setEditing(null);
    if (node) {
      node.show();
      node.getLayer()?.batchDraw();
    }
  }, [editing]);

  const updateEditingValue = useCallback(
    (value: string, region: TypographyRegion) => {
      setEditing((prev) => (prev ? { ...prev, value } : prev));
      const node = nodesRef.current.get(region.region_id);
      if (node) previewFit(region, node, value);
    },
    [previewFit]
  );

  return {
    editing,
    beginEdit,
    commitEdit,
    cancelEdit,
    updateEditingValue,
    nodesRef,
  };
}

/**
 * TypographyEditOverlay
 * ----------------------
 * The HTML <textarea> counterpart rendered by the host component while
 * `editing` is non-null. Positioned in screen space over the hidden
 * Konva.Text node; styled to match the source typography exactly so the
 * transition between "reading" and "editing" is visually seamless.
 *
 * Usage in host component:
 *
 *   {editing && (
 *     <TypographyEditOverlay
 *       state={editing}
 *       region={regions.find(r => r.region_id === editing.regionId)!}
 *       onChange={(v) => updateEditingValue(v, region)}
 *       onCommit={commitEdit}
 *       onCancel={cancelEdit}
 *     />
 *   )}
 */
export interface TypographyEditOverlayProps {
  state: EditingState;
  region: TypographyRegion;
  onChange: (value: string) => void;
  onCommit: () => void;
  onCancel: () => void;
}
