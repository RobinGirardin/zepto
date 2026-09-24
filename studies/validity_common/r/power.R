#!/usr/bin/env Rscript
# Sample size from the largest of the eight cell SDs.
#
# Framework §6. After a protocol-matched pilot, enter the largest
# phase × precision × hypothesis SD of subject-level relative error as
# sigma and solve
#
#   power_t_TOST(type = "one.sample", eqb = 0.10, alpha = 0.05,
#                power = 0.9936, delta = 0, sd = sigma)
#
# n is the number of subjects, not CUDA rows and not training steps.

args <- commandArgs(trailingOnly = TRUE)

evaluation_path <- NULL
sigma_override <- NULL
out_path <- NULL

i <- 1
while (i <= length(args)) {
  if (args[[i]] == "--evaluation" && i < length(args)) {
    evaluation_path <- args[[i + 1]]
    i <- i + 2
  } else if (args[[i]] == "--sigma" && i < length(args)) {
    sigma_override <- as.numeric(args[[i + 1]])
    i <- i + 2
  } else if (args[[i]] == "--out" && i < length(args)) {
    out_path <- args[[i + 1]]
    i <- i + 2
  } else {
    stop("unknown argument: ", args[[i]], call. = FALSE)
  }
}

if (!requireNamespace("TOSTER", quietly = TRUE)) {
  stop("Install TOSTER: install.packages(\"TOSTER\")", call. = FALSE)
}

relative_error <- function(y, t) {
  if (any(t == 0)) {
    stop("twin probe t must be non-zero", call. = FALSE)
  }
  (y - t) / t
}

cell_sds <- NULL
if (is.null(sigma_override)) {
  if (is.null(evaluation_path)) {
    stop("pass --evaluation PATH or --sigma VALUE", call. = FALSE)
  }
  rows <- read.csv(evaluation_path, stringsAsFactors = FALSE)
  required <- c("subject_id", "phase", "precision", "y_vram", "target_vram",
                "y_flop_fcm", "target_flop")
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop("evaluation.csv missing columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }

  rows$re_vram <- relative_error(rows$y_vram, rows$target_vram)
  rows$re_flop <- relative_error(rows$y_flop_fcm, rows$target_flop)

  # One relative error per completed subject in each phase × precision cell.
  cells <- list()
  for (phase in c("inference", "training")) {
    for (precision in c("fp16", "fp32")) {
      part <- rows[rows$phase == phase & rows$precision == precision, ]
      part <- part[!duplicated(part$subject_id), ]
      cells[[paste("vram", phase, precision, sep = ".")]] <- sd(part$re_vram)
      cells[[paste("flop", phase, precision, sep = ".")]] <- sd(part$re_flop)
    }
  }
  cell_sds <- unlist(cells)
  if (any(is.na(cell_sds))) {
    stop("at least one cell has fewer than two completed subjects", call. = FALSE)
  }
  sigma <- max(cell_sds)
} else {
  sigma <- sigma_override
}

power <- TOSTER::power_t_TOST(
  n = NULL,
  delta = 0,
  sd = sigma,
  eqb = 0.10,
  alpha = 0.05,
  power = 0.9936,
  type = "one.sample"
)

n_subjects <- as.integer(ceiling(power$n))

result <- list(
  n_subjects = n_subjects,
  n_raw = unname(power$n),
  sigma = unname(sigma),
  eqb = 0.10,
  alpha = 0.05,
  power = 0.9936,
  delta = 0,
  type = "one.sample",
  cell_sds = if (is.null(cell_sds)) NA else as.list(cell_sds)
)

cat("Largest cell SD (sigma):", format(sigma, digits = 6), "\n")
if (!is.null(cell_sds)) {
  cat("Eight cell SDs:\n")
  print(cell_sds)
}
cat("power_t_TOST n (raw):", format(power$n, digits = 6), "\n")
cat("Subjects to collect (ceiling):", n_subjects, "\n")

if (!is.null(out_path)) {
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  json <- sprintf(
    "{\n  \"n_subjects\": %d,\n  \"n_raw\": %s,\n  \"sigma\": %s,\n  \"eqb\": 0.10,\n  \"alpha\": 0.05,\n  \"power\": 0.9936,\n  \"delta\": 0\n}\n",
    n_subjects,
    format(power$n, digits = 12, scientific = FALSE),
    format(sigma, digits = 12, scientific = FALSE)
  )
  writeLines(json, out_path)
  cat("Wrote", out_path, "\n")
}
