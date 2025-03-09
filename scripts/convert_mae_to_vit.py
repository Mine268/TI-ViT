from typing import *
from transformers import ViTMAEModel, ViTModel, ViTConfig
import torch
import logging


def convert_vit_mae_to_vit(
    mae_model: Union[str, ViTMAEModel],
    output_path: str = "./converted_vit_model",
    verify: bool = True,
    **kwargs
) -> ViTModel:
    """
    将ViTMAEModel转换为标准ViTModel

    参数:
        mae_model: 支持以下两种输入方式
            - str: Hugging Face模型名称或本地路径(例如 "facebook/vit-mae-base")
            - ViTMAEModel实例: 已加载的MAE模型对象
        output_path: 转换后模型的保存路径
        verify: 是否验证输出一致性
        **kwargs: 传递给ViTConfig的额外参数

    返回:
        ViTModel: 转换后的标准ViT模型
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 加载原始模型
        if isinstance(mae_model, str):
            logging.info(f"Loading MAE model from {mae_model}")
            mae_model = ViTMAEModel.from_pretrained(mae_model)
        elif not isinstance(mae_model, ViTMAEModel):
            raise TypeError("mae_model必须是模型路径或ViTMAEModel实例")

        # 创建ViT配置 (自动过滤MAE特有参数)
        mae_config = mae_model.config
        vit_config_args = {
            k: v for k, v in mae_config.to_dict().items()
            if not k.startswith('decoder') and k != 'mask_ratio'
        }
        vit_config_args.update(kwargs)
        vit_config = ViTConfig(**vit_config_args)

        # 初始化目标模型
        logging.info("Initializing ViT model")
        vit_model = ViTModel(vit_config)

        # 参数迁移
        def transfer_parameters(src_module, dst_module, log_name=""):
            src_params = dict(src_module.named_parameters())
            for name, param in dst_module.named_parameters():
                # 处理ViTMAE的layernorm后缀差异
                # normalized_name = name.replace('.ln_', '.norm.')
                normalized_name = name
                if normalized_name in src_params:
                    param.data.copy_(src_params[normalized_name].data)
                elif name in src_params:
                    param.data.copy_(src_params[name].data)
                else:
                    logging.warning(f"{log_name}: 参数 {name} 未找到对应项")

        # 迁移关键组件
        logging.info("Transferring parameters...")

        # 迁移embeddings
        transfer_parameters(mae_model.embeddings, vit_model.embeddings, "Embeddings")

        # 迁移encoder
        transfer_parameters(mae_model.encoder, vit_model.encoder, "Encoder")

        # 迁移layernorm
        transfer_parameters(mae_model.layernorm, vit_model.layernorm, "LayerNorm")

        # 验证输出
        if verify:
            logging.info("Verifying output consistency with mask filtering...")
            dummy_input = torch.randn(1, 3, mae_config.image_size, mae_config.image_size)

            # 生成MAE的mask索引
            with torch.no_grad():
                # 获取patch嵌入
                mae_embeddings = mae_model.embeddings(dummy_input)  # [1, 197, 768]
                num_patches = mae_config.image_size // mae_config.patch_size
                num_patches = num_patches ** 2

                # 模拟MAE的随机mask生成
                mask_ratio = 0.75
                len_keep = int(num_patches * (1 - mask_ratio))
                noise = torch.rand(1, num_patches, device=dummy_input.device)
                ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列，小值在前
                ids_keep = ids_shuffle[:, :len_keep]  # [1, len_keep]

                # 构建MAE编码器输入
                cls_token = mae_embeddings[:, :1, :]  # CLS token
                visible_patches = mae_embeddings[:, 1:, :][0, ids_keep[0]].unsqueeze(0)
                mae_encoder_input = torch.cat([cls_token, visible_patches], dim=1)

                # MAE编码器输出
                mae_output = mae_model.encoder(mae_encoder_input).last_hidden_state

            # ViT完整输出
            with torch.no_grad():
                vit_output = vit_model(dummy_input).last_hidden_state  # [1, 197, 768]

            # 筛选ViT输出
            vit_cls = vit_output[:, :1, :]  # CLS token
            vit_visible = vit_output[:, 1:, :][0, ids_keep[0]].unsqueeze(0)  # 可见patch
            selected_vit_output = torch.cat([vit_cls, vit_visible], dim=1)

            # 验证形状和数值
            assert mae_output.shape == selected_vit_output.shape, \
                f"Shape mismatch: {mae_output.shape} vs {selected_vit_output.shape}"

            mse = torch.mean((mae_output - selected_vit_output) ** 2)
            logging.info(f"Masked Output MSE: {mse.item():.2e}")

            if mse < 1e-6:
                logging.info("验证通过 ✅")
            else:
                logging.warning("关键差异警告 ⚠️ 请检查："
                                "1.位置嵌入参数 2.patch投影层 3.层归一化参数")

        # 保存模型
        if output_path:
            logging.info(f"Saving model to {output_path}")
            vit_model.save_pretrained(output_path)

        return vit_model

    except Exception as e:
        logging.error(f"转换失败: {str(e)}")
        raise


if __name__ == "__main__":
    convert_vit_mae_to_vit(
        "./models/facebook/vit-mae-base",
        "./models/facebook/converted-vit-base",
        verify=False)  # manually checked yes