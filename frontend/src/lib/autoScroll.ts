/** 自动滚动跟随的判定阈值（像素）：视口底边距内容底部不超过它，
 * 就认为用户"停在底部"，继续跟随新内容；一旦上翻远离，跟随即让位。 */
export const SCROLL_FOLLOW_THRESHOLD_PX = 80;

/** 视口底边是否贴近滚动内容底部。参数直接取自滚动容器的
 * scrollTop / clientHeight / scrollHeight（window 场景用
 * scrollY / innerHeight / documentElement.scrollHeight），便于调用与测试。 */
export function isNearBottom(
  scrollTop: number,
  clientHeight: number,
  scrollHeight: number,
  threshold: number = SCROLL_FOLLOW_THRESHOLD_PX,
): boolean {
  return scrollHeight - (scrollTop + clientHeight) <= threshold;
}
