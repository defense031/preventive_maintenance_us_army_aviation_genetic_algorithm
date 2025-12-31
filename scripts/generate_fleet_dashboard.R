#!/usr/bin/env Rscript
# generate_fleet_dashboard.R
# Fleet Analysis Dashboard for aviation_genetic_v1
#
# PURPOSE: Create executive-level fleet performance visualizations
# USAGE: Rscript scripts/generate_fleet_dashboard.R --db-path results/.../simulation_data.db
# OUTPUT: 4-panel dashboard + individual plots saved to visualizations/

cat("📊 FLEET ANALYSIS DASHBOARD - Aviation Genetic v1\n")
cat("================================================\n\n")

# Load required packages
suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(gridExtra)
})

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

#' Calculate daily OR trend with readiness categories
#' @param daily_data Daily aircraft data from database
#' @return Data frame with OR metrics by day
calculate_fleet_or_trend <- function(daily_data) {
  or_trend <- daily_data %>%
    group_by(day) %>%
    summarise(
      fmc_count = sum(status == "FMC"),
      total_aircraft = n(),
      or = fmc_count / total_aircraft,
      or_category = case_when(
        or >= 0.85 ~ "R1+",  # Dark green
        or >= 0.75 ~ "R1",   # Green
        or >= 0.60 ~ "R2",   # Yellow
        or >= 0.50 ~ "R3",   # Orange
        TRUE ~ "R4"          # Red
      ),
      .groups = "drop"
    )

  return(or_trend)
}

# ===================================================================
# PLOT GENERATION FUNCTIONS
# ===================================================================

#' Plot 1: Fleet Operational Readiness Over Time
#' @param daily_data Daily aircraft data
#' @param title_prefix Optional title prefix
#' @return ggplot object
plot_or_trend <- function(daily_data, title_prefix = "") {
  or_trend <- calculate_fleet_or_trend(daily_data)

  p <- ggplot(or_trend, aes(x = day, y = or)) +
    geom_line(linewidth = 1, color = "steelblue") +
    geom_point(aes(color = or_category), size = 2) +
    geom_hline(yintercept = 0.75, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_color_manual(
      values = c("R1+" = "darkgreen", "R1" = "green", "R2" = "yellow",
                 "R3" = "orange", "R4" = "red"),
      name = "Readiness"
    ) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    labs(
      title = paste0(title_prefix, "Fleet Operational Readiness Over Time"),
      subtitle = "Target: 75% OR (dashed line)",
      x = "Simulation Day",
      y = "Operational Readiness (%)"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  return(p)
}

#' Plot 2: Fleet RUL Distribution Over Time
#' @param daily_data Daily aircraft data
#' @param title_prefix Optional title prefix
#' @param sample_interval Sample every N days (default: 30 for monthly)
#' @return ggplot object
plot_fleet_rul <- function(daily_data, title_prefix = "", sample_interval = 30) {
  fleet_rul <- daily_data %>%
    filter(day %% sample_interval == 0) %>%
    group_by(day) %>%
    summarise(
      avg_rul = mean(true_rul, na.rm = TRUE),
      min_rul = min(true_rul, na.rm = TRUE),
      max_rul = max(true_rul, na.rm = TRUE),
      q25_rul = quantile(true_rul, 0.25, na.rm = TRUE),
      q75_rul = quantile(true_rul, 0.75, na.rm = TRUE),
      .groups = "drop"
    )

  p <- ggplot(fleet_rul, aes(x = day)) +
    geom_ribbon(aes(ymin = min_rul, ymax = max_rul), alpha = 0.2, fill = "gray") +
    geom_ribbon(aes(ymin = q25_rul, ymax = q75_rul), alpha = 0.4, fill = "steelblue") +
    geom_line(aes(y = avg_rul), color = "darkblue", linewidth = 1) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "red", alpha = 0.7) +
    labs(
      title = paste0(title_prefix, "Fleet RUL Distribution Over Time"),
      subtitle = "Dark line: average, Blue band: IQR, Gray band: full range",
      x = "Simulation Day",
      y = "RUL (Hours)"
    ) +
    theme_minimal()

  return(p)
}

#' Plot 3: Fleet Maintenance Events Over Time
#' @param daily_data Daily aircraft data
#' @param title_prefix Optional title prefix
#' @return ggplot object
plot_maintenance_events <- function(daily_data, title_prefix = "") {
  # Use state transition approach for accurate event counting
  maintenance_events <- daily_data %>%
    arrange(aircraft_id, day) %>%
    group_by(aircraft_id) %>%
    mutate(
      prev_status = lag(status, default = "FMC"),
      maintenance_started = ifelse(
        !grepl("maintenance_", prev_status) & grepl("maintenance_", status),
        status, NA
      )
    ) %>%
    filter(!is.na(maintenance_started)) %>%
    ungroup() %>%
    mutate(
      week = floor(day / 7) + 1,
      maintenance_type = case_when(
        maintenance_started == "maintenance_reactive" ~ "Reactive",
        maintenance_started == "maintenance_preventive" ~ "Preventive",
        maintenance_started == "maintenance_minor_phase" ~ "Minor Phase",
        maintenance_started == "maintenance_major_phase" ~ "Major Phase",
        TRUE ~ "Other"
      )
    )

  # Count events for subtitle
  if (nrow(maintenance_events) > 0) {
    event_counts <- maintenance_events %>%
      count(maintenance_type) %>%
      arrange(desc(n))

    subtitle_text <- paste(
      "Total Events:",
      paste(paste0(event_counts$maintenance_type, ": ", event_counts$n), collapse = ", ")
    )

    p <- ggplot(maintenance_events, aes(x = week, fill = maintenance_type)) +
      geom_histogram(binwidth = 1, alpha = 0.8, position = "stack") +
      scale_fill_manual(
        values = c(
          "Reactive" = "red",
          "Preventive" = "steelblue",
          "Minor Phase" = "orange",
          "Major Phase" = "purple",
          "Other" = "gray"
        ),
        name = "Type"
      ) +
      labs(
        title = paste0(title_prefix, "Fleet Maintenance Events Over Time"),
        subtitle = subtitle_text,
        x = "Week",
        y = "Number of Events"
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
  } else {
    p <- ggplot() +
      ggtitle("No maintenance events found") +
      theme_void()
  }

  return(p)
}

#' Plot 4: Fleet Performance - OR vs Flight Hours
#' @param daily_data Daily aircraft data
#' @param title_prefix Optional title prefix
#' @return ggplot object
plot_performance_analysis <- function(daily_data, title_prefix = "") {
  pareto_data <- daily_data %>%
    group_by(day) %>%
    summarise(
      daily_or = mean(status == "FMC"),
      total_flight_hours = sum(todays_flight_hours, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      efficiency = daily_or / (total_flight_hours + 1),  # +1 to avoid division by zero
      time_period = case_when(
        day <= max(day) * 0.33 ~ "Early",
        day <= max(day) * 0.67 ~ "Middle",
        TRUE ~ "Late"
      )
    )

  p <- ggplot(pareto_data, aes(x = total_flight_hours, y = daily_or)) +
    geom_point(aes(color = time_period, size = efficiency), alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, color = "red", alpha = 0.3) +
    scale_color_manual(
      values = c("Early" = "green", "Middle" = "orange", "Late" = "blue"),
      name = "Time Period"
    ) +
    scale_size_continuous(name = "Efficiency", guide = "legend") +
    scale_y_continuous(labels = percent_format()) +
    labs(
      title = paste0(title_prefix, "Fleet Performance: OR vs Flight Hours"),
      subtitle = "Size indicates efficiency (OR per flight hour)",
      x = "Daily Fleet Flight Hours",
      y = "Daily Operational Readiness (%)"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  return(p)
}

# ===================================================================
# DATA EXTRACTION
# ===================================================================

#' Load daily data from SQLite database
#' @param db_path Path to simulation_data.db
#' @param episode_id Optional episode ID (defaults to last episode)
#' @return Data frame with daily aircraft state
load_daily_data <- function(db_path, episode_id = NULL) {
  cat(sprintf("📂 Connecting to database: %s\n", basename(db_path)))

  # Connect to database
  conn <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(conn), add = TRUE)

  # Get last episode if not specified
  if (is.null(episode_id)) {
    # Try both column names for compatibility
    result <- tryCatch({
      dbGetQuery(conn, "SELECT MAX(episode) as max_ep FROM daily_data")
    }, error = function(e) {
      dbGetQuery(conn, "SELECT MAX(episode_id) as max_ep FROM daily_data")
    })

    episode_id <- result$max_ep[1]
    cat(sprintf("📊 Using last episode: %d\n", episode_id))
  }

  # Extract daily data for specified episode
  daily_data <- tryCatch({
    dbGetQuery(conn, "SELECT * FROM daily_data WHERE episode = ?", params = list(episode_id))
  }, error = function(e) {
    dbGetQuery(conn, "SELECT * FROM daily_data WHERE episode_id = ?", params = list(episode_id))
  })

  if (nrow(daily_data) == 0) {
    stop(sprintf("❌ No data found for episode %d", episode_id))
  }

  cat(sprintf("✅ Loaded %d rows for episode %d\n", nrow(daily_data), episode_id))
  cat(sprintf("   Days: %d, Aircraft: %d\n",
              max(daily_data$day),
              length(unique(daily_data$aircraft_id))))

  return(daily_data)
}

# ===================================================================
# DASHBOARD GENERATION
# ===================================================================

#' Generate complete fleet dashboard
#' @param db_path Path to simulation_data.db
#' @param episode_id Optional episode ID (defaults to last episode)
#' @param output_dir Output directory for plots
#' @param session_name Session name for plot titles
#' @param save_plots Whether to save plots to files
#' @return List with all plots and summary statistics
generate_fleet_dashboard <- function(db_path, episode_id = NULL,
                                    output_dir = NULL,
                                    session_name = "",
                                    save_plots = TRUE) {

  cat("\n🚀 Starting fleet dashboard generation...\n\n")

  # Load data
  daily_data <- load_daily_data(db_path, episode_id)

  # Create title prefix
  title_prefix <- if (session_name != "") paste0(session_name, ": ") else ""

  # Generate all 4 plots
  cat("📊 Generating visualizations...\n")

  plots <- list(
    or_trend = plot_or_trend(daily_data, title_prefix),
    fleet_rul = plot_fleet_rul(daily_data, title_prefix),
    maintenance_timeline = plot_maintenance_events(daily_data, title_prefix),
    performance_analysis = plot_performance_analysis(daily_data, title_prefix)
  )

  cat(sprintf("✅ Generated %d plots\n", length(plots)))

  # Calculate summary statistics
  summary_stats <- list(
    simulation_days = max(daily_data$day),
    total_aircraft = length(unique(daily_data$aircraft_id)),
    avg_or = mean(daily_data$status == "FMC"),
    min_or = min(daily_data %>% group_by(day) %>% summarise(or = mean(status == "FMC")) %>% pull(or)),
    max_or = max(daily_data %>% group_by(day) %>% summarise(or = mean(status == "FMC")) %>% pull(or)),
    total_flight_hours = sum(daily_data$todays_flight_hours, na.rm = TRUE)
  )

  # Print summary
  cat("\n📋 SUMMARY STATISTICS\n")
  cat("====================\n")
  cat(sprintf("Simulation Length: %d days\n", summary_stats$simulation_days))
  cat(sprintf("Fleet Size: %d aircraft\n", summary_stats$total_aircraft))
  cat(sprintf("Average OR: %.1f%%\n", summary_stats$avg_or * 100))
  cat(sprintf("OR Range: %.1f%% - %.1f%%\n",
              summary_stats$min_or * 100,
              summary_stats$max_or * 100))
  cat(sprintf("Total Flight Hours: %.0f hours\n", summary_stats$total_flight_hours))

  # Save plots if requested
  if (save_plots) {
    # Determine output directory
    if (is.null(output_dir)) {
      # Extract from database path
      db_dir <- dirname(db_path)
      output_dir <- file.path(db_dir, "visualizations")
    }

    # Create output directory
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
      cat(sprintf("\n📁 Created output directory: %s\n", output_dir))
    }

    cat(sprintf("\n💾 Saving plots to: %s\n", output_dir))

    # Save individual plots
    for (plot_name in names(plots)) {
      filename <- file.path(output_dir, paste0("fleet_dashboard_", plot_name, ".png"))
      ggsave(filename, plots[[plot_name]], width = 12, height = 8, dpi = 300)
      cat(sprintf("  ✅ Saved: %s\n", basename(filename)))
    }

    # Create combined 2x2 dashboard
    tryCatch({
      combined_plot <- grid.arrange(
        plots$or_trend,
        plots$fleet_rul,
        plots$maintenance_timeline,
        plots$performance_analysis,
        ncol = 2, nrow = 2,
        top = paste("Fleet Analysis Dashboard -", session_name)
      )

      combined_filename <- file.path(output_dir, "fleet_dashboard_combined.png")
      ggsave(combined_filename, combined_plot, width = 16, height = 12, dpi = 300)
      cat(sprintf("  ✅ Combined dashboard: %s\n", basename(combined_filename)))
    }, error = function(e) {
      cat("⚠️  Could not create combined dashboard:", e$message, "\n")
    })
  }

  cat("\n✅ Fleet dashboard generation complete!\n")

  return(list(
    plots = plots,
    summary_stats = summary_stats,
    daily_data = daily_data
  ))
}

# ===================================================================
# COMMAND LINE INTERFACE
# ===================================================================

if (!interactive()) {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  # Simple argument parsing
  db_path <- NULL
  output_dir <- NULL
  episode_id <- NULL
  session_name <- ""

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--db-path" && i < length(args)) {
      db_path <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output-dir" && i < length(args)) {
      output_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--episode-id" && i < length(args)) {
      episode_id <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--session-name" && i < length(args)) {
      session_name <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--help") {
      cat("Fleet Dashboard Generator for aviation_genetic_v1\n\n")
      cat("Usage:\n")
      cat("  Rscript scripts/generate_fleet_dashboard.R --db-path <path> [options]\n\n")
      cat("Required:\n")
      cat("  --db-path <path>        Path to simulation_data.db\n\n")
      cat("Optional:\n")
      cat("  --output-dir <path>     Output directory (default: <db_dir>/visualizations)\n")
      cat("  --episode-id <int>      Episode ID to visualize (default: last episode)\n")
      cat("  --session-name <name>   Session name for plot titles\n")
      cat("  --help                  Show this help message\n\n")
      cat("Example:\n")
      cat("  Rscript scripts/generate_fleet_dashboard.R \\\n")
      cat("    --db-path results/test_validation.../simulation_data.db \\\n")
      cat("    --session-name \"Baseline Test\"\n\n")
      quit(status = 0)
    } else {
      i <- i + 1
    }
  }

  # Validate required arguments
  if (is.null(db_path)) {
    cat("❌ Error: --db-path is required\n")
    cat("Run with --help for usage information\n")
    quit(status = 1)
  }

  if (!file.exists(db_path)) {
    cat(sprintf("❌ Error: Database file not found: %s\n", db_path))
    quit(status = 1)
  }

  # Generate dashboard
  result <- generate_fleet_dashboard(
    db_path = db_path,
    episode_id = episode_id,
    output_dir = output_dir,
    session_name = session_name,
    save_plots = TRUE
  )

  cat("\n🎉 Dashboard generation complete!\n")
}
