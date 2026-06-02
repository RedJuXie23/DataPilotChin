declare module 'plotly.js/dist/plotly' {
  export function relayout(graphDiv: any, layoutUpdate: Record<string, any>): Promise<void>

  export function downloadImage(
    graphDiv: any,
    options: {
      format: string
      filename: string
      width: number
      height: number
      scale: number
    }
  ): Promise<void>
}
