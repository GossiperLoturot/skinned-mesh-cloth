using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

public class SkinnedMeshCloth : MonoBehaviour
{
    public ComputeShader shader;

    private RenderGraph _renderGraph;
    private SkinnedMeshClothPass _pass;

    void OnEnable()
    {
        _renderGraph = new RenderGraph("Skinned Mesh Cloth Graph");
        _pass = new SkinnedMeshClothPass(shader);

        RenderPipelineManager.beginCameraRendering += OnBeginCamera;
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCamera;

        _pass.Dispose();
    }

    private void OnBeginCamera(ScriptableRenderContext ctx, Camera cam)
    {
        var cmd = CommandBufferPool.Get("Skinned Mesh Cloth Command");
        var param = new RenderGraphParameters
        {
            scriptableRenderContext = ctx,
            commandBuffer = cmd,
            currentFrameIndex = Time.frameCount,
        };

        try
        {
            _renderGraph.BeginRecording(param);
            _pass.RecordRenderGraph(_renderGraph);
            _renderGraph.EndRecordingAndExecute();
        }
        catch (System.Exception e)
        {
            _renderGraph.Cleanup();
            throw e;
        }

        ctx.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
        ctx.Submit();
    }
}

class PassData
{
    public ComputeShader shader;

    public int count;
    public BufferHandle srcBuffer;
    public BufferHandle dstBuffer;
}

class SkinnedMeshClothPass : System.IDisposable
{
    private ComputeShader _shader;

    private int _count;
    private GraphicsBuffer _srcBuffer;
    private GraphicsBuffer _dstBuffer;

    public SkinnedMeshClothPass(ComputeShader shader)
    {
        if (shader == null) throw new System.ArgumentNullException(nameof(shader));

        _shader = shader;

        _count = 256;
        _srcBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, 4);
        _dstBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, 4);
    }

    public void RecordRenderGraph(RenderGraph renderGraph)
    {
        using (var builder = renderGraph.AddComputePass<PassData>("Skinned Mesh Flow Pass", out var passData))
        {
            passData.shader = _shader;

            passData.count = 256;
            passData.srcBuffer = renderGraph.ImportBuffer(_srcBuffer);
            passData.dstBuffer = renderGraph.ImportBuffer(_dstBuffer);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                // compute flow
                var kernelId = passData.shader.FindKernel("CSMain");
                ctx.cmd.SetComputeIntParam(passData.shader, "Count", passData.count);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "SrcBuffer", passData.srcBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "DstBuffer", passData.dstBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, 1, 1, 1);
            });
        }
    }

    public void Dispose()
    {
        _srcBuffer.Dispose();
        _dstBuffer.Dispose();
    }
}