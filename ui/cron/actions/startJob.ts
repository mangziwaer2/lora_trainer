import prisma from '../prisma';
import { Job } from '@prisma/client';
import { execFileSync, spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken } from '../paths';

const isWindows = process.platform === 'win32';
const STARTUP_GRACE_MS = 10000;

const LEGACY_TOOLKIT_PREFIXES = [
  '/workspace/ai-toolkit',
  '\\workspace\\ai-toolkit',
];

const normalizeLegacyToolkitPath = (value: unknown): unknown => {
  if (typeof value !== 'string' || value.trim() === '') {
    return value;
  }

  const normalizedValue = value.replace(/\\/g, '/');
  const legacyPrefix = LEGACY_TOOLKIT_PREFIXES.find(prefix => normalizedValue.startsWith(prefix.replace(/\\/g, '/')));
  if (!legacyPrefix) {
    return value;
  }

  const relativePath = normalizedValue.slice(legacyPrefix.replace(/\\/g, '/').length).replace(/^\/+/, '');
  return relativePath ? path.join(TOOLKIT_ROOT, relativePath) : TOOLKIT_ROOT;
};

const normalizeJobConfigPaths = (jobConfig: any, trainingRoot: string) => {
  const processConfig = jobConfig?.config?.process?.[0];
  if (!processConfig) {
    return jobConfig;
  }

  processConfig.training_folder = trainingRoot;
  processConfig.sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

  if (Array.isArray(processConfig.datasets)) {
    for (const dataset of processConfig.datasets) {
      dataset.folder_path = normalizeLegacyToolkitPath(dataset.folder_path);
      dataset.dataset_path = normalizeLegacyToolkitPath(dataset.dataset_path);
      dataset.mask_path = normalizeLegacyToolkitPath(dataset.mask_path);
      dataset.control_path = normalizeLegacyToolkitPath(dataset.control_path);
      dataset.control_path_1 = normalizeLegacyToolkitPath(dataset.control_path_1);
      dataset.control_path_2 = normalizeLegacyToolkitPath(dataset.control_path_2);
      dataset.control_path_3 = normalizeLegacyToolkitPath(dataset.control_path_3);
      dataset.clip_image_path = normalizeLegacyToolkitPath(dataset.clip_image_path);
    }
  }

  if (processConfig.model) {
    processConfig.model.name_or_path = normalizeLegacyToolkitPath(processConfig.model.name_or_path);
    processConfig.model.assistant_lora_path = normalizeLegacyToolkitPath(processConfig.model.assistant_lora_path);
  }

  if (Array.isArray(processConfig.sample?.samples)) {
    for (const sample of processConfig.sample.samples) {
      sample.ctrl_img = normalizeLegacyToolkitPath(sample.ctrl_img);
      sample.ctrl_img_1 = normalizeLegacyToolkitPath(sample.ctrl_img_1);
      sample.ctrl_img_2 = normalizeLegacyToolkitPath(sample.ctrl_img_2);
      sample.ctrl_img_3 = normalizeLegacyToolkitPath(sample.ctrl_img_3);
    }
  }

  return jobConfig;
};

const appendLogLine = (logPath: string, message: string) => {
  try {
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
  } catch (error) {
    console.error('Error writing startup log line:', error);
  }
};

const getSelectedGpuMemoryMb = (gpuIds: string): number | null => {
  try {
    const output = execFileSync(
      'nvidia-smi',
      ['--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
      {
        encoding: 'utf8',
        windowsHide: true,
      },
    );

    const memoryByGpu = new Map<string, number>();
    for (const line of output.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      const [gpuIndex, memoryMb] = trimmed.split(',').map(part => part.trim());
      const parsedMemory = Number(memoryMb);
      if (!Number.isNaN(parsedMemory)) {
        memoryByGpu.set(gpuIndex, parsedMemory);
      }
    }

    const selectedGpuIds = gpuIds
      .split(',')
      .map(id => id.trim())
      .filter(Boolean);

    if (selectedGpuIds.length === 0) {
      return null;
    }

    const selectedMemory = selectedGpuIds
      .map(id => memoryByGpu.get(id))
      .filter((value): value is number => typeof value === 'number');

    if (selectedMemory.length !== selectedGpuIds.length) {
      return null;
    }

    return Math.min(...selectedMemory);
  } catch {
    return null;
  }
};

const clampDimensionToMultipleOf64 = (value: number) => {
  return Math.max(64, Math.round(value / 64) * 64);
};

const applyLowVramSamplingSafety = (jobConfig: any, gpuIds: string, logPath: string) => {
  const processConfig = jobConfig?.config?.process?.[0];
  if (!processConfig) {
    return jobConfig;
  }

  const selectedGpuMemoryMb = getSelectedGpuMemoryMb(gpuIds);
  if (selectedGpuMemoryMb === null || selectedGpuMemoryMb > 6144) {
    return jobConfig;
  }

  const notes: string[] = [`Detected low-VRAM GPU (${selectedGpuMemoryMb} MiB)`];

  if (processConfig.model?.arch === 'sdxl' && processConfig.model.low_vram !== true) {
    processConfig.model.low_vram = true;
    notes.push('enabled model.low_vram');
  }

  if (processConfig.train?.disable_sampling === true || !processConfig.sample) {
    appendLogLine(logPath, `${notes.join('; ')}. Sampling safety unchanged because sampling is disabled.`);
    return jobConfig;
  }

  if (!processConfig.train?.force_first_sample && processConfig.train?.skip_first_sample !== true) {
    processConfig.train.skip_first_sample = true;
    notes.push('enabled train.skip_first_sample');
  }

  const sampleWidth = Number(processConfig.sample.width);
  const sampleHeight = Number(processConfig.sample.height);
  if (!Number.isNaN(sampleWidth) && !Number.isNaN(sampleHeight) && sampleWidth > 0 && sampleHeight > 0) {
    const maxPixels = 512 * 512;
    const samplePixels = sampleWidth * sampleHeight;
    if (samplePixels > maxPixels) {
      const scale = Math.sqrt(maxPixels / samplePixels);
      const clampedWidth = clampDimensionToMultipleOf64(sampleWidth * scale);
      const clampedHeight = clampDimensionToMultipleOf64(sampleHeight * scale);
      if (clampedWidth !== sampleWidth || clampedHeight !== sampleHeight) {
        processConfig.sample.width = clampedWidth;
        processConfig.sample.height = clampedHeight;
        notes.push(`clamped sample resolution ${sampleWidth}x${sampleHeight} -> ${clampedWidth}x${clampedHeight}`);
      }
    }
  }

  const sampleSteps = Number(processConfig.sample.sample_steps);
  if (!Number.isNaN(sampleSteps) && sampleSteps > 20) {
    processConfig.sample.sample_steps = 20;
    notes.push(`reduced sample_steps ${sampleSteps} -> 20`);
  }

  appendLogLine(logPath, notes.join('; '));
  return jobConfig;
};

const markJobLaunchFailed = async (jobID: string, info: string, pidPath?: string) => {
  try {
    await prisma.job.updateMany({
      where: {
        id: jobID,
        status: 'running',
        info: 'Starting job...',
      },
      data: {
        status: 'error',
        info,
      },
    });
  } catch (error) {
    console.error('Error updating failed launch status:', error);
  }

  if (!pidPath) {
    return;
  }

  try {
    if (fs.existsSync(pidPath)) {
      fs.unlinkSync(pidPath);
    }
  } catch (error) {
    console.error('Error removing pid file after failed launch:', error);
  }
};

const startAndWatchJob = (job: Job) => {
  return new Promise<void>(async resolve => {
    const jobID = job.id;
    const trainingRoot = await getTrainingFolder();
    const trainingFolder = path.join(trainingRoot, job.name);

    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    const configPath = path.join(trainingFolder, '.job_config.json');
    const logPath = path.join(trainingFolder, 'log.txt');
    const pidPath = path.join(trainingFolder, 'pid.txt');

    try {
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        let num = 0;
        while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
          num++;
        }

        fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
      }
    } catch (error) {
      console.error('Error moving log file:', error);
    }

    const jobConfig = applyLowVramSamplingSafety(
      normalizeJobConfigPaths(JSON.parse(job.job_config), trainingRoot),
      job.gpu_ids,
      logPath,
    );
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    let pythonPath = 'python';
    if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
      pythonPath = isWindows
        ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
        : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
    } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
      pythonPath = isWindows
        ? path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe')
        : path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
    }

    const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
    if (!fs.existsSync(runFilePath)) {
      console.error(`run.py not found at path: ${runFilePath}`);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: 'Error launching job: run.py not found',
        },
      });
      return;
    }

    const additionalEnv: Record<string, string> = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
    };

    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    const args = [runFilePath, configPath, '--log', logPath];
    appendLogLine(logPath, `Launching job ${jobID}`);
    appendLogLine(logPath, `Python: ${pythonPath}`);
    appendLogLine(logPath, `Args: ${args.map(arg => JSON.stringify(arg)).join(' ')}`);

    try {
      const logFd = fs.openSync(logPath, 'a');
      let logFdClosed = false;
      let launchWatchActive = true;
      const startupTimer = setTimeout(() => {
        launchWatchActive = false;
      }, STARTUP_GRACE_MS);
      startupTimer.unref?.();

      const closeLogFd = () => {
        if (logFdClosed) {
          return;
        }
        try {
          fs.closeSync(logFd);
          logFdClosed = true;
        } catch (error) {
          console.error('Error closing launch log handle:', error);
        }
      };

      const subprocess = spawn(pythonPath, args, {
        env: {
          ...process.env,
          ...additionalEnv,
        },
        cwd: TOOLKIT_ROOT,
        detached: true,
        windowsHide: isWindows,
        stdio: ['ignore', logFd, logFd],
      });

      subprocess.once('error', async error => {
        clearTimeout(startupTimer);
        closeLogFd();
        appendLogLine(logPath, `Failed to spawn training process: ${error.message}`);
        await markJobLaunchFailed(jobID, `Error launching job: ${error.message}`, pidPath);
      });

      subprocess.once('exit', async (code, signal) => {
        clearTimeout(startupTimer);
        closeLogFd();
        if (!launchWatchActive) {
          return;
        }

        const exitParts = [];
        if (code !== null) {
          exitParts.push(`code ${code}`);
        }
        if (signal) {
          exitParts.push(`signal ${signal}`);
        }
        const exitSummary = exitParts.length > 0 ? exitParts.join(', ') : 'unknown exit';
        appendLogLine(logPath, `Training process exited during startup (${exitSummary})`);
        await markJobLaunchFailed(jobID, `Training process exited during startup (${exitSummary}). See log.txt`, pidPath);
      });

      if (subprocess.unref) {
        subprocess.unref();
      }

      try {
        fs.writeFileSync(pidPath, String(subprocess.pid ?? ''), { flag: 'w' });
        appendLogLine(logPath, `Spawned training process with PID ${subprocess.pid ?? 'unknown'}`);
      } catch (error) {
        console.error('Error writing pid file:', error);
      }

      resolve();
    } catch (error: any) {
      console.error('Error launching process:', error);
      appendLogLine(logPath, `Error launching process: ${error?.message || 'Unknown error'}`);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
    }
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }

  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      info: 'Starting job...',
    },
  });

  startAndWatchJob(job);
}
