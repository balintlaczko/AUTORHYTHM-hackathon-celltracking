# plot coordinates over time 

data_with_ids <- data_with_ids %>%
  mutate(unique_id = as.factor(unique_id))

data_with_ids %>%
  filter(timepoint == 2) %>%
  ggplot(aes(x, y, group = unique_id, label = unique_id)) +
  geom_point(aes(fill = unique_id, size = Cells_AreaShape_Area),alpha = 0.1) +
  theme_minimal() +
  guides(fill = FALSE, size = FALSE) +
  scale_y_reverse() +  # Invert the y scale
  geom_text(size = 3, position = position_nudge(y = 0.05))  # Adjust the position for text labels
           

library(gganimate)
library(gifski)


# Create an animation by grouping by timepoint
animation <- data_with_ids %>%
  ggplot(aes(x, y, group = unique_id, label = unique_id)) +
  geom_point(aes(fill = unique_id), size = 5, alpha = 0.1) +
  theme_minimal() +
  guides(fill = FALSE, size = FALSE) +
  scale_y_reverse() +  # Invert the y scale
  #geom_text(size = 3, position = position_nudge(y = 0.05)) +  # Adjust the position for text labels
  labs(title = 'Timepoint: {frame_timepoint}', x = 'x coordinates', y = 'y coordinates') +
  transition_states(timepoint, transition_length = 1, state_length = 1) +
  ease_aes('linear')
  #enter_fade() +
  #exit_fade()

# Create an animated gif using the gifski backend
anim <- animate(animation, nframes = 100, renderer = gifski_renderer())  # You can adjust nframes as needed

anim

# Create an animated gif
anim_save("animated_plot.gif", animation)
