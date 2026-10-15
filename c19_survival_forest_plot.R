
###############################################################################
# SECTION 1: Load required libraries
###############################################################################
library(ggplot2)
library(readxl)
library(gridExtra)
library(grid)
library(cowplot)
library(dplyr)
library(ggtext)
library(latex2exp)


###############################################################################
# SECTION 2: Import data and results
###############################################################################

# df is a dataframe comprising variables names, units, SHAP-derived HRs + 95% CIs, reference points for SHAP-derived HRs, number of entries per reference point, and p-values from the Python XAI script

# df <- df %>% add_row(
#   Variable = ...,
#   Level = ...,
#   HR = ...,
#   CI_lower = ...,
#   CI_upper = ...,
#   p_value = ...,
#   N_upper = ..., 
#   N_lower = ...
# )


# Sort variables by HR
df <- df %>%
  arrange(HR) %>%
  mutate(Variable = factor(Variable, levels = unique(Variable)))

###############################################################################
# SECTION 3: Helper Function for Forest Plot Formatting
###############################################################################

# Function: make_forest_rows
# Description: Formats HR and CI for text display, assigns row indices for plotting
make_forest_rows <- function(df) {
  df %>%
    mutate(
      HR_CI = sprintf("%.2f (%.2f–%.2f)", HR, CI_lower, CI_upper),
      p_val_fmt = format.pval(p_value, digits = 3, eps = .001),
      row_id = row_number(),
      N_levels_fmt = if_else(
        is.na(N_lower) | is.na(N_upper),
        "",
        paste0(
          "(",
          as.integer(N_lower),
          ", ",
          as.integer(N_upper),
          ")"
        )
      ),
    )
}

# Apply the function
df_plot <- make_forest_rows(df)

###############################################################################
# SECTION 4: Define logarithmic HR axis
###############################################################################

# Display range for the logarithmic HR axis
hr_limits <- c(0.5, 3.5)

# Tick marks shown on the logarithmic axis
hr_breaks <- c(0.5, 0.75, 1, 1.5, 2, 3, 3.5)

# Maximum two decimal places, without unnecessary trailing zeros
format_hr <- function(x) {
  sub("\\.?0+$", "", sprintf("%.2f", x))
}
###############################################################################
# SECTION 5: Build Forest Plot Components
###############################################################################

# Variable labels
label_plot <- ggplot(df_plot, aes(y = row_id)) +
  geom_richtext(aes(x = 0, label = Variable), hjust = 0, size = 4.2,
                fill = NA, label.color = NA) +
  theme_void() + xlim(0, 1)

# Comparison level labels
level_plot <- ggplot(df_plot, aes(y = row_id)) +
  geom_text(
    aes(
      x = 0,
      label = latex2exp::TeX(
        dplyr::coalesce(Level, ""),
        output = "character"
      )
    ),
    parse = TRUE,
    hjust = 0,
    size = 4.2,
    na.rm = TRUE
  ) +
  theme_void() +
  xlim(0, 1)

# Sample-size counts at the two comparison levels
n_levels_plot <- ggplot(
  df_plot,
  aes(y = row_id)
) +
  geom_text(
    aes(
      x = 0,
      label = N_levels_fmt
    ),
    hjust = 0,
    size = 4.2,
    na.rm = TRUE
  ) +
  theme_void() +
  xlim(0, 1)

# HR and CI text
hr_ci_plot <- ggplot(df_plot, aes(y = row_id)) +
  geom_text(aes(x = 0, label = HR_CI), hjust = 0, size = 4.2) +
  theme_void() + xlim(0, 1)

# P-value text
pval_plot <- ggplot(df_plot, aes(y = row_id)) +
  geom_text(aes(x = 0, label = p_val_fmt), hjust = 0, size = 4.2) +
  theme_void() + xlim(0, 1)

###############################################################################
# Forest plot with x-axis directly below the forest panel
###############################################################################

forest_plot <- ggplot(
  df_plot,
  aes(x = HR, y = row_id)
) +
  geom_errorbarh(
    aes(
      xmin = CI_lower,
      xmax = CI_upper
    ),
    height = 0.2,
    na.rm = TRUE
  ) +
  geom_point(
    size = 1.5,
    na.rm = TRUE
  ) +
  geom_vline(
    xintercept = 1,
    linetype = "dashed",
    color = "gray50"
  ) +
  
  # Logarithmic HR axis
  scale_x_log10(
    name = "HR",
    breaks = hr_breaks,
    minor_breaks = NULL,
    labels = format_hr,
    expand = expansion(mult = c(0, 0))
  ) +
  
  # Display only HR 0.5–3.5, WITHOUT removing observations
  coord_cartesian(
    xlim = hr_limits,
    clip = "on"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    
    axis.text.x = element_text(
      size = 10,
      color = "black"
    ),
    axis.ticks.x = element_line(
      color = "black",
      linewidth = 0.4
    ),
    axis.ticks.length.x = unit(3, "pt"),
    axis.line.x = element_line(
      color = "black",
      linewidth = 0.4
    ),
    axis.title.x = element_text(
      size = 12,
      margin = margin(t = 8)
    ),
    
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    
    plot.margin = margin(
      t = 5,
      r = 5,
      b = 5,
      l = 5
    )
  )
###############################################################################
# SECTION 6: Combine Panels into Final Forest Plot
###############################################################################

# Title helper function
make_title <- function(label, size = 5) {
  ggplot() +
    annotate(
      "text",
      x = 0,
      y = 0.5,
      label = label,
      hjust = 0,
      vjust = 0.5,
      size = size,
      fontface = "bold",
      lineheight = 0.9
    ) +
    coord_cartesian(
      xlim = c(0, 1),
      ylim = c(0, 1),
      clip = "off"
    ) +
    theme_void()
}

# Define column widths and layout
widths_used <- c(
  0.55, # Variable
  0.28, # Level
  0.38, # N 
  0.45, # Forest plot
  0.40, # HR (95% CI)
  0.15  # p-value
)

ncol_used <- 6

# Header row
header_row <- arrangeGrob(
  make_title("Variable"),
  make_title("Level"),
  make_title(
    "N (lower level,\nupper level)",
    size = 4.2
  ),
  make_title("Forest plot"),
  make_title("HR (95% CI)"),
  make_title("p-value"),
  ncol = ncol_used,
  widths = widths_used
)

# Align all body components
aligned_body_plots <- cowplot::align_plots(
  label_plot,
  level_plot,
  n_levels_plot,
  forest_plot,
  hr_ci_plot,
  pval_plot,
  align = "h",
  axis = "tb"
)

# Arrange the six body columns
body_row <- arrangeGrob(
  grobs = aligned_body_plots,
  ncol = ncol_used,
  widths = widths_used
)

# Combine header and body
final_plot <- arrangeGrob(
  header_row,
  body_row,
  ncol = 1,
  heights = c(0.11, 1)
)

###############################################################################
# SECTION 7: Export Final Plot
###############################################################################

# Save to SVG format for high-quality vector output
svg("name_for_saving_forest_plot.svg", width = 11, height = 5)
grid.draw(final_plot)
dev.off()

grid.newpage()
grid.draw(final_plot)
