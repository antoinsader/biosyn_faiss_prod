



import datetime
import  json, os, torch, time, logging
try:
    import psutil
except ImportError:
    psutil = None

from config import GlobalConfig



# ====================
# MY LOGGER
# ====================

class MyLogger:
    def __init__(self, logger, current_experiment_log_path, logger_tag):
        """
            Args:
                logger: from logger lib, it will create a file where all the logger.info will be saved
                current_experiment_log_path: the log file path (set in checkpointing)
                logger_tag: will use this tag when we print our logs (could be train - eval -...)


            Example of the class usage:
                lg = MyLogger(logger, './logs/my_log.log', 'train')
                lg.log_event('Faiss search finished', message=f'FAISS RECALL@20 = 0.9', t0=faiss_search_time_start, log_memory=True, epoch=1)
            Will log the message:
                 ======================================================================
                ▶ EVENT LOG :: [TRAIN] :: [Faiss search finished] :: epoch=1
                ----------------------------------------------------------------------
                        Meesage: FAISS RECALL@20 = 0.9
                        Event elapsed time: 2 seconds
                        CPU memory: 1024 MB
                        GPU MEMORY STATS: free/total = {free:.1f}/{total:.1f}MB - allocated/peak allocated = {allocated:.1f}/{allocated_peak:.1f}MB - reserved/peak reserved = {reserved:.1f}/{reserved_peak:.1f}MB"

                --------------------------------------------------------------------------
                ■ EVENT LOG END :: [Faiss search finished]
                ======================================================================
        """
        self.logger = logger
        self.current_experiment_log_path = current_experiment_log_path
        self.logger_tag = logger_tag


        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        if psutil:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

        self._set_log_file_to_logger()




    def _set_log_file_to_logger(self):
        fmt = logging.Formatter('%(message)s')
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self.logger.addHandler(console)
        file_handler = logging.FileHandler(self.current_experiment_log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)


    def current_cpu_mem_usage(self):
        """
            Return the rss of the cpu ram
        """
        if self.process:
            rss = self.process.memory_info().rss / (1024 ** 2)
            return rss
        return 0.0


    def current_gpu_mem_usage(self):
        """
            Return tuple (free, total) for the gpu ram in the moment of the call
        """
        if self.use_cuda:
            free = torch.cuda.mem_get_info()[0] / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return (free, total)
        return (0.0,0.0)


    def current_gpu_stats(self):
        """
            Return (allocated, allocated_peak, reserved, reserved_peak) in MB
            allocated: Memory currently allocated by tensors.
            allocated_peak: Highest memory allocated by tensors since the program start or last reset.
            reserved: Memory reserved by the caching allocator (includes allocated plus cached blocks).
            reserved_peak: Highest reserved memory since the program start or last reset.
        """

        if not self.use_cuda:
            return (None, None, None, None)
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        allocated_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        reserved_peak = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        return (allocated, allocated_peak, reserved, reserved_peak)


    def current_gpu_stats_str(self):
        """
            Return a message ready to be logged for the gpu memory usage with the step number
        """
        (free, total) = self.current_gpu_mem_usage()
        (allocated, allocated_peak, reserved, reserved_peak) = self.current_gpu_stats()
        return f"GPU MEMORY STATS: free/total = {free:.1f}/{total:.1f}MB - allocated/peak allocated = {allocated:.1f}/{allocated_peak:.1f}MB - reserved/peak reserved = {reserved:.1f}/{reserved_peak:.1f}MB"




    def log_event(
        self,
        event_tag,
        message=None,
        t0=None,
        log_memory=True,
        epoch=None,
        ):
        """
            Log an event in the shape:
                ======================================================================
                ▶ EVENT LOG :: [<logger_tag>] :: [<event_tag>] :: epoch=<epoch>
                ----------------------------------------------------------------------
                        Meesage: <message>   (if message is passed)
                        Event elapsed time: <time> seconds   (if t0 -time where the event started- is passed)
                        CPU memory: <cpu_ram>MB (if log_memory is passed)
                        GPU memory: <GPU MEMORY DETAILS> (if log_memory is passed)
                --------------------------------------------------------------------------
                ■ EVENT LOG END :: [<event_tag>]
                ======================================================================
        """

        # HEADER
        header = f"[{self.logger_tag}] :: [{event_tag}]"
        if epoch is not None:
            header += f" :: epoch= {epoch}"

        # Build message body
        lines = []
        if message:
            lines.append(f"Message     : {message} \n")
        if t0:
            elapsed = time.time() - t0
            lines.append(f"Event elapsed time: {elapsed:.2f} seconds")

        if log_memory:
            cpu_memory = f"{self.current_cpu_mem_usage():.1f} MB"
            lines.append(f"CPU Memory  : {cpu_memory}")

            if self.use_cuda:
                gpu_message = self.current_gpu_stats_str()
                lines.append(gpu_message)

        border = "=" * 70
        dash_line = "-" * 70
        formatted = (
            f"\n{border}\n"
            f"▶ EVENT LOG :: {header}\n"
            f"\n{dash_line}\n"
            + "\t\t"
            + "\n\t\t".join(lines)
            + f"\n{dash_line}\n"
            f"■ EVENT LOG END :: {event_tag}\n"
            f"{border}\n"
        )
        return self.logger.info(formatted)



# =======================
#       CHECKPOINTING
#========================
class CheckPointing:
    def __init__(self,  cfg:GlobalConfig, eval:bool=False):
        """
            The class is responsible for:
                1- Write into global log file which having all the previous experiments:
                            Each run is a new experiment (train  + eval), and each experiment has a unique experience_id 
                2- Checkpointing the model training:
                            While you're doing a training, if the training stopped for some reason, 
                            this class is responsible for restoring last checkpoint of the model and 
                            continuing the training from the last epoch arrived in the interrupted experience
                            This is done using the global log file and the experience_id which would be set to {finished: false}
                            unless the experiment is finished
                3- Set the output dir:
                            The output dir is where the encoder and faiss index are saved at the end of the experiment
                            This class would create this dir based on the experience_id and output_dir set in conf
                            So for example if the experience_id is 1, the trained files would be saved in '/output/encoder_1/'
                4- At the end of the training part of the experience, this class would write information about the experience:
                            So in the end you would have global log file containing json array, 
                            inside the array are the objects of experiments giving summary of the experiment
                            The details are: 
                                    id: unique experiment id
                                    finished: if the experiment is finished training, end_date: timestamp when the training finished
                                    training_log_name: just a unique name to recognize the experiment,
                                    queries_size, dictionary_size, trained_period, 
                                    log_details_file (path of the file having the logs),
                                    num epochs, 
                                    last faiss recall@15, last average accuracy@5,
                                    last average mrr, last average loss
                                    result encoder dir (where the output is saved for this experiment) 
        """

        self.cfg = cfg
        self.eval = eval
        self.logs_dir =  cfg.paths.logs_dir
        self.global_log_path = cfg.paths.global_log_path

        self.current_experiment_log_path = None
        self.current_experiment_id = None

        self.current_experiment, self.all_experiments = self._get_last_global_experiment()


        assert self.current_experiment_id is not None
        assert self.current_experiment_log_path is not None
        assert self.current_experiment["id"] == self.all_experiments[-1]["id"]

        if not self.eval:
            cfg.paths.set_result_encoder_dir(cfg.paths.output_dir + f"/encoder_{self.current_experiment_id}/")
            self.checkpoint_path = cfg.paths.checkpoint_path
            assert self.checkpoint_path is not None

    def _get_last_global_experiment(self):
        """
           Reading all experiments from the global log file
           If there is unfinished experiment, make it the current experiment
           Otherwise create a new experiment with new id and mark it with finished = false 
        """

        all_experiments = []
        if not os.path.isfile(self.global_log_path):
            with open(self.global_log_path, "w") as f:
                json.dump(all_experiments,f)
        os.makedirs(self.logs_dir, exist_ok=True)


        with open(self.global_log_path, "r") as f:
            all_experiments = json.load(f)


        if self.eval:
            # eval
            result_encoder_dir = self.cfg.paths.result_encoder_dir

            exps = [x for x in all_experiments if  x.get("result_encoder_dir") == result_encoder_dir]

            if len(exps) != 1:
                self.current_experiment_id = all_experiments[-1]["id"] + 1 if len(all_experiments) > 0 else 1
                datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.current_experiment_log_path = self.logs_dir + f"/log_eval_{self.current_experiment_id}_{datestr}.log"
                current_experiment = {
                    "id": self.current_experiment_id,
                    "eval_started": True,
                    "log_details_file": self.current_experiment_log_path                    
                }
                all_experiments.append(current_experiment)
            else:
                self.current_experiment_id = exps[-1]["id"]
                self.current_experiment_log_path = exps[-1]['log_details_file']
                exps[-1]["eval_started"] = True
                assert exps[-1].get("finished", False), f"The experiment of {result_encoder_dir} has not finished training"
                current_experiment = exps[-1]


        else:
            # train
            unfinished = [x for x in all_experiments if not x.get("finished", False)]
            if unfinished:
                current_experiment = unfinished[-1]
                self.current_experiment_id = current_experiment['id']
                self.current_experiment_log_path = current_experiment['log_details_file']
                current_experiment = current_experiment
            else:
                self.current_experiment_id = all_experiments[-1]["id"] + 1 if len(all_experiments) > 0 else 1
                datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.current_experiment_log_path = self.logs_dir + f"/log_{self.current_experiment_id}_{datestr}.log"

                current_experiment = {
                    "id": self.current_experiment_id,
                    "start_time": datestr,
                    "train_started": True,
                    "finished": False,
                    "training_log_name": self.cfg.logger.train_log_name,
                    "log_details_file": self.current_experiment_log_path
                }
                all_experiments.append(current_experiment)

        with open(self.global_log_path, "w") as f:
            json.dump(all_experiments, f, indent=2)

        return current_experiment, all_experiments

    def log_finished(self, queries_len, dict_len, training_time_str, last_acc_5, last_mrr, last_faiss_recall, last_loss ):
        """
            This function would be called when the training is finished
            Log into the global log file a summary about the current experiment and mark it as finished
        """

        self.current_experiment["finished"] = True
        self.current_experiment["end_date"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.current_experiment["training_log_name"] = self.cfg.logger.train_log_name
        self.current_experiment["queries_size"] = queries_len
        self.current_experiment["dictionary_size"] = dict_len
        self.current_experiment["trained_period"] = training_time_str
        self.current_experiment["log_details_file"] = self.current_experiment_log_path
        self.current_experiment["epochs"] = self.cfg.train.num_epochs
        self.current_experiment["last_faiss_recall@15"] = last_faiss_recall
        self.current_experiment["last_avg_acc_5"] = last_acc_5
        self.current_experiment["last_avg_mrr"] = last_mrr
        self.current_experiment["last_avg_loss"] = last_loss
        self.current_experiment["result_encoder_dir"] = self.cfg.paths.result_encoder_dir

        self.all_experiments[-1] = self.current_experiment
        with open(self.global_log_path, "w") as f:
            json.dump(self.all_experiments, f, indent=2)
        return True

    def save_checkpoint(self, chkpt):
        torch.save(chkpt, self.checkpoint_path)
        return True

    def restore_checkpoint(self):
        chkpt = torch.load(self.checkpoint_path)
        return chkpt
