library(dplyr)
library(tidyr)
library(purrr)
library(here)

# Assuming your data is already loaded and named 'data'
# and it has columns: 'id', 'timepoint', 'x', 'y'

# import data

data_df <- read.csv(here("hackathon_dataset_S2cells_231124.csv"))

# Assume df is your data frame with columns: time, x, y
df <- data_df %>%
  filter(Image_Metadata_Well == "N17" &
           Image_Metadata_Multipoint == 3) %>%
  rename(x = Cells_Location_Center_X,
         y = Cells_Location_Center_Y,
         # time = timepoint
  ) %>%
  arrange(timepoint) %>%
  group_by(timepoint) %>%
  mutate(id = row_number()) %>%
  ungroup()

data <- df %>%
  filter(!Image_Metadata_Condition == "BST")

calculate_distance <- function(x1, y1, x2, y2) {
  sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

# Function to find closest objects
find_closest_objects <- function(data) {
  results <- list()
  
  unique_timepoints <- unique(data$timepoint)
  for (i in seq_len(length(unique_timepoints) - 1)) {
    current_tp <- unique_timepoints[i]
    next_tp <- unique_timepoints[i + 1]
    
    current_data <- filter(data, timepoint == current_tp)
    next_data <- filter(data, timepoint == next_tp)
    
    distances <- expand.grid(current_id = current_data$id, next_id = next_data$id) %>%
      mutate(distance = mapply(calculate_distance,
                               current_data$x[current_id], current_data$y[current_id],
                               next_data$x[next_id], next_data$y[next_id]))
    
    closest <- distances %>%
      group_by(current_id) %>%
      summarise(closest_id = next_id[which.min(distance)], .groups = 'drop') %>%
      mutate(timepoint = current_tp)
    
    results[[i]] <- closest
  }
  
  bind_rows(results)
}

closest_links <- find_closest_objects(data)

# Initialize a column for unique identifiers
data <- mutate(data, unique_id = NA_integer_)



# Add unique identifiers to dataframe -------------------------------------


library(dplyr)
library(tidyr)
library(purrr)

# Assuming 'data' has columns: 'id', 'timepoint', 'x', 'y'
# Assuming 'calculate_distance' function is already defined
# Assuming 'closest_links' has been generated correctly

# Assign initial unique identifiers for the first timepoint
first_timepoint <- min(data$timepoint)
data <- data %>%
  mutate(unique_id = if_else(timepoint == first_timepoint, as.integer(row_number()), NA_integer_))

# Initialize the next unique identifier to be assigned
next_unique_id <- max(data$unique_id[data$timepoint == first_timepoint]) + 1

# Function to propagate unique identifiers
propagate_identifiers <- function(data, links, next_unique_id) {
  for (tp in unique(data$timepoint)[-1]) { # Skip the first timepoint
    current_links <- filter(links, timepoint == tp - 1)
    
    for (i in seq_len(nrow(current_links))) {
      link <- current_links[i, ]
      source_id <- link$current_id
      target_id <- link$closest_id
      
      # Propagate the identifier if the source object has one
      if (!is.na(data$unique_id[data$id == source_id & data$timepoint == tp - 1])) {
        data$unique_id[data$id == target_id & data$timepoint == tp] <- data$unique_id[data$id == source_id & data$timepoint == tp - 1]
      } else {
        # Assign a new identifier if the source object does not have one
        data$unique_id[data$id == target_id & data$timepoint == tp] <- next_unique_id
        next_unique_id <- next_unique_id + 1
      }
    }
    
    # Assign new identifiers to unlinked objects in this timepoint
    unlinked_ids <- setdiff(data$id[data$timepoint == tp], data$id[data$timepoint == tp - 1][current_links$current_id])
    for (obj_id in unlinked_ids) {
      if (is.na(data$unique_id[data$id == obj_id & data$timepoint == tp])) {
        data$unique_id[data$id == obj_id & data$timepoint == tp] <- next_unique_id
        next_unique_id <- next_unique_id + 1
      }
    }
  }
  list(data = data, next_unique_id = next_unique_id)
}

result <- propagate_identifiers(data, closest_links, next_unique_id)
data_with_ids <- result$data

# Check the results
head(data_with_ids)



# plot cell area over time for each unique_id

data_with_ids %>%
  ggplot(aes(timepoint, Cells_AreaShape_Area, group= unique_id)) +
  geom_line() + 
  theme_minimal()

write.csv(here(data_with_ids), "data_with_ids.csv")

# 
# 
# 
# # Assign unique identifiers
# unique_identifier <- 1
# for (tp in unique(data$timepoint)) {
#   # Propagate identifiers
#   if (tp < max(data$timepoint)) {
#     links <- filter(closest_links, timepoint == tp)
#     for (i in seq_len(nrow(links))) {
#       linked_id <- links$closest_id[i]
#       data$unique_id[data$id == linked_id & data$timepoint == tp + 1] <- data$unique_id[data$id == links$current_id[i] & data$timepoint == tp]
#     }
#   }
#   
#   # Identify untracked objects at the current timepoint
#   untracked <- setdiff(data$id[data$timepoint == tp], links$current_id)
#   
#   for (obj_id in untracked) {
#     data$unique_id[data$id == obj_id & data$timepoint == tp] <- unique_identifier
#     unique_identifier <- unique_identifier + 1
#   }
# }
# 
# # Checking the results
# head(data)
