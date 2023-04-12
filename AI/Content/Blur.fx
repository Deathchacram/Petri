#if OPENGL
	#define SV_POSITION POSITION
	#define VS_SHADERMODEL vs_3_0
	#define PS_SHADERMODEL ps_3_0
#else
	#define VS_SHADERMODEL vs_4_0_level_9_1
	#define PS_SHADERMODEL ps_4_0_level_9_1
#endif

Texture2D SpriteTexture;
float2 Suv;
float4 Color;

sampler2D SpriteTextureSampler = sampler_state
{
	Texture = <SpriteTexture>;
};

struct VertexShaderOutput
{
	float4 Position : SV_POSITION;
	float4 Color : COLOR0;
	float2 TextureCoordinates : TEXCOORD0;
};

float4 MainPS(VertexShaderOutput input) : COLOR
{
    //float2 suv = float2(1.0f / 400.0f, 1.0f / 225.0f);
    float2 suv = Suv;
    float2 uv = input.TextureCoordinates;
    float4 col = tex2D(SpriteTextureSampler, uv);
    if (col.r == 0 && col.g == 0 && col.b == 0)
    {
        float3 gr = saturate(tex2D(SpriteTextureSampler, uv + suv.xy).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv - suv.xy).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv - float2(0, suv.y)).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv + float2(suv.x, 0)).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv - float2(suv.x, 0)).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv + float2(0, suv.y)).rgb)
        + saturate(tex2D(SpriteTextureSampler, uv + float2(suv.x, -suv.y)).rgb)
		+ saturate(tex2D(SpriteTextureSampler, uv + float2(-suv.x, suv.y)).rgb);
        col.rgb = col.rgb + gr / 6;
    }
    return col;
	//return tex2D(SpriteTextureSampler,input.TextureCoordinates) * input.Color;
}

technique SpriteDrawing
{
	pass P0
	{
		PixelShader = compile PS_SHADERMODEL MainPS();
	}
};