import { GroupedSelectOption, SelectOption } from '@/types';

type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections =
  | 'model.quantize'
  | 'train.timestep_type'
  | 'network.conv'
  | 'trigger_word'
  | 'train.diff_output_preservation'
  | 'train.blank_prompt_preservation'
  | 'train.unload_text_encoder'
  | 'slider';

type AdditionalSections =
  | 'datasets.control_path'
  | 'datasets.multi_control_paths'
  | 'datasets.do_i2v'
  | 'sample.ctrl_img'
  | 'sample.multi_ctrl_imgs'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path';

type ModelGroup = 'image' | 'instruction' | 'video';

export interface ModelArch {
  name: string;
  label: string;
  group: ModelGroup;
  controls?: Control[];
  isVideoModel?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
  accuracyRecoveryAdapters?: { [key: string]: string };
}

const defaultNameOrPath = '';

export const modelArchs: ModelArch[] = [
  {
    name: 'sdxl',
    label: 'SDXL',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['stabilityai/stable-diffusion-xl-base-1.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].sample.sampler': ['ddpm', 'ddpm'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'ddpm'],
      'config.process[0].sample.guidance_scale': [6, 6],
      'config.process[0].sample.num_frames': [1, 1],
      'config.process[0].sample.fps': [1, 1],
      'config.process[0].datasets[0].num_frames': [1, 1],
      'config.process[0].datasets[0].do_i2v': [false, false],
      'config.process[0].datasets[0].shrink_video_to_frames': [false, false],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
];

export const groupedModelOptions: GroupedSelectOption[] = [
  {
    label: 'image',
    options: [{ value: 'sdxl', label: 'SDXL' }],
  },
];

export const quantizationOptions: SelectOption[] = [
  { value: '', label: '- NONE -' },
  { value: 'qfloat8', label: 'float8 (unused for SDXL)' },
];

export const defaultQtype = 'qfloat8';

interface JobTypeOption extends SelectOption {
  disableSections?: DisableableSections[];
}

export const jobTypeOptions: JobTypeOption[] = [
  {
    value: 'diffusion_trainer',
    label: 'LoRA Trainer',
    disableSections: ['slider'],
  },
];
