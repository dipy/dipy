import ui
import window
ls = ui.RangeSlider()
sm=window.ShowManager(size=(600,600))
sm.ren.add(ls)
sm.start()