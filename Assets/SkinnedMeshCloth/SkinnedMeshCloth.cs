using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.VFX;

public class SkinnedMeshCloth : MonoBehaviour
{
    public ComputeShader shader;
    public VisualEffect visualEffect;

    private RenderGraph _renderGraph;
    private SkinnedMeshClothPass _pass;

    void OnEnable()
    {
        _renderGraph = new RenderGraph("Skinned Mesh Cloth Graph");
        _pass = new SkinnedMeshClothPass(shader, visualEffect);

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
    public BufferHandle idxBuffer;
}

class SkinnedMeshClothPass : System.IDisposable
{
    private ComputeShader _shader;
    private VisualEffect _visualEffect;

    private GraphicsBuffer _idxBuffer;
    private bool _computed;

    private int COUNT = 1024;
    private const int THREADNUM = 64;

    public SkinnedMeshClothPass(ComputeShader shader, VisualEffect visualEffect)
    {
        if (shader == null) throw new System.ArgumentNullException(nameof(shader));

        _shader = shader;
        _visualEffect = visualEffect;

        _idxBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, COUNT, 4);

        var data = new float[COUNT];
        for (int i = 0; i < COUNT; i++) data[i] = Random.value;
        _idxBuffer.SetData(data);
    }

    public void RecordRenderGraph(RenderGraph renderGraph)
    {
        if (_computed) return;

        using (var builder = renderGraph.AddComputePass<PassData>("Skinned Mesh Cloth Pass", out var passData))
        {
            passData.shader = _shader;

            passData.count = COUNT;
            passData.idxBuffer = renderGraph.ImportBuffer(_idxBuffer);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                var kernelId = passData.shader.FindKernel("CSMain");

                int nlog = 0;
                for (int n = passData.count; n > 1; n >>= 1) nlog++;

                for (int i = 0; i < nlog; i++)
                {
                    int inc = 1 << i;
                    for (int j = 0; j < i + 1; j++)
                    {
                        ctx.cmd.SetComputeIntParam(passData.shader, "inc", inc);
                        ctx.cmd.SetComputeIntParam(passData.shader, "dir", 2 << i);
                        ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "data", passData.idxBuffer);

                        int threadGroups = (passData.count >> 1) / THREADNUM;
                        ctx.cmd.DispatchCompute(passData.shader, kernelId, threadGroups, 1, 1);

                        inc >>= 1;
                    }
                }
            });
        }

        // DEBUG
        _visualEffect.SetInt("Count", COUNT);
        _visualEffect.SetGraphicsBuffer("IdxBuffer", _idxBuffer);
        _visualEffect.Play();

        _computed = true;
    }

    public void Dispose()
    {
        _idxBuffer.Dispose();
    }
}